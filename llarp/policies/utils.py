#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import abc
import os.path as osp
import warnings
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from habitat import logger
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer, LlamaConfig,
                          LlamaForCausalLM, LlamaModel, LlamaTokenizer,
                          T5Model)

from llarp.policies.llama_parallel import LlamaModelParallel
from llarp.task.utils import get_parser


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class LlmWrapper(nn.Module, abc.ABC):
    @abc.abstractmethod
    def decode(
        self,
        non_causal_tokens,
        causal_embeds,
        att_masks,
        kv_cache=None,
        use_cache=False,
        n_non_causal_tokens=None,
    ):
        pass

    @property
    @abc.abstractmethod
    def base_model(self) -> nn.Module:
        pass

    @property
    @abc.abstractmethod
    def llm_head(self) -> nn.Module:
        pass

    @property
    @abc.abstractmethod
    def d_model(self) -> int:
        pass

    def get_vision_output_shape(self):
        return None

    def get_visual_pooler(self):
        return None

    def get_visual_encoder(self):
        return None

    def get_image_processor(self):
        return None

    @property
    @abc.abstractmethod
    def config(self):
        pass


def freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False


class DecoderWrapper(LlmWrapper):
    def __init__(
        self,
        llm_id,
        use_b16: bool,
        model_cfg: Dict[str, Any],
        peft_settings,
        peft,
        peft_full_att_params,
        train_llm: bool = False,
        load_in_8bit=False,
        use_rope_scaling=False,
        debug_mode=False,
        model_parallel_factor: int = 1,
        force_non_causal: bool = False,
        **kwargs,
    ):
        """
        :param force_non_causal: Legacy option to load models trained without
            the CausalLM HF model type.
        """

        super().__init__()
        self._debug_mode = debug_mode
        if self._debug_mode:
            self._tokenizer = get_parser(llm_id)
        else:
            self._tokenizer = None

        logger.info(f"Loading LLM from {llm_id}")

        kwargs = {}
        if use_b16:
            kwargs = {"torch_dtype": torch.bfloat16}

        cmp_llm_id = llm_id.lower()
        if llm_id is None or llm_id == "":
            # Load with a custom config.
            self.llm = LlamaForCausalLM(LlamaConfig(**model_cfg))
        elif "llama" in cmp_llm_id:
            rope_scaling = (
                {"type": "dynamic", "factor": 2} if use_rope_scaling else None
            )
            self.llm = LlamaForCausalLM.from_pretrained(
                llm_id,
                load_in_8bit=load_in_8bit,
                rope_scaling=rope_scaling,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported LLM type {llm_id}")

        if use_b16:
            logger.info(f"Setting model data type to bfloat16.")
            create_type = torch.bfloat16
            self.llm.base_model.to(create_type)
        else:
            create_type = torch.float32

        if force_non_causal:
            self.llm = self.llm.base_model

        self._create_type = create_type
        self._model_parallel_factor = model_parallel_factor
        self._non_causal_mask = None
        self.tmp_skip_debug = False
        self._train_llm = train_llm
        self._peft = peft

        if self._peft:
            self.llm = setup_peft_module(self.llm, peft_full_att_params, peft_settings)
        elif not self._train_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        logger.info(f"Done loading LLM")

    def parameters(self):
        if self._peft or self._train_llm:
            return self.llm.parameters()
        else:
            return []

    @property
    def config(self):
        return self.llm.config

    @property
    def base_model(self):
        if self._peft:
            # Skip past the PEFT wrapper.
            return self.llm.base_model.base_model
        elif hasattr(self.llm, "base_model"):
            return self.llm.base_model
        else:
            return self.llm

    @property
    def llm_head(self) -> nn.Module:
        return self.llm.lm_head

    @property
    def d_model(self):
        return self.llm.config.hidden_size

    def _verify_input(self, non_causal_tokens, causal_embeds, att_masks):
        if self.tmp_skip_debug:
            return
        debug_info = causal_embeds[..., -2:]
        # Check each env individually.
        for env_prefix_tokens, env_debug_info, env_att_masks in zip(
            non_causal_tokens, debug_info, att_masks
        ):
            env_att_masks = env_att_masks.view(-1)
            valid_debug_info = env_debug_info[env_att_masks].int()
            step_idx = valid_debug_info[:, 0]
            ep_idx = valid_debug_info[:, 1]

            if (ep_idx[0] != ep_idx).all():
                raise ValueError(f"The episode indices don't match {ep_idx}")

            n_attend = env_att_masks.sum()
            if step_idx.max() + 1 != n_attend:
                raise ValueError(
                    f"Attention mask is wrong {env_att_masks}, compare to steps {step_idx}"
                )
            step_diff = step_idx[1:] - step_idx[:-1]
            if not (len(step_diff) == 0 or (step_diff == 1).all()):
                raise ValueError(f"Steps are inconsistent {step_diff}")

            text = self._tokenizer.decode(env_prefix_tokens)

    def embed_tokens(self, tokens) -> torch.Tensor:
        return self.base_model.embed_tokens(tokens)

    def decode(
        self,
        non_causal_tokens,
        causal_embeds,
        att_masks,
        kv_cache=None,
        use_cache=False,
        n_non_causal_tokens=None,
    ):
        if self._debug_mode:
            self._verify_input(non_causal_tokens, causal_embeds, att_masks)
            # Remove the debug info.
            causal_embeds = causal_embeds[..., :-2]

        if non_causal_tokens is None:
            all_embeds = causal_embeds
        else:
            causal_embeds = causal_embeds.to(self._create_type)
            non_causal_embeds = self.embed_tokens(non_causal_tokens)
            all_embeds = torch.cat([non_causal_embeds, causal_embeds], dim=1)

            non_causal_mask = self._get_non_causal_mask(non_causal_tokens.shape[1])

        if n_non_causal_tokens is None:
            n_non_causal_tokens = non_causal_tokens.shape[1]
        non_causal_mask = self._get_non_causal_mask(n_non_causal_tokens)
        non_causal_mask = non_causal_mask.expand(all_embeds.shape[0], -1)
        att_masks = att_masks.view(*att_masks.shape[:2])
        att_masks = torch.cat([non_causal_mask, att_masks], dim=-1)

        assert len(att_masks.shape) == 2

        if use_cache:
            kwargs = dict(
                use_cache=use_cache,
                past_key_values=kv_cache,
            )
        else:
            kwargs = {}

        seq_out = self.base_model.forward(
            inputs_embeds=all_embeds, attention_mask=att_masks.int(), **kwargs
        )

        # Ignore the part of the sequence from the prefix
        context_len = causal_embeds.shape[1]
        return (
            seq_out.last_hidden_state[:, -context_len:].to(torch.float32),
            seq_out.get("past_key_values", None),
        )

    def to(self, device):
        self.llm.to(device)
        self._device = device
        return self

    def _get_non_causal_mask(self, seq_len):
        # Recalculate the non-causal mask if the input shape change from the
        # last iteration or if the non-causal mask hasn't been created yet.
        if self._non_causal_mask is None or self._non_causal_mask.shape[1] != seq_len:
            # Cache this mask so we don't have to reallocate.
            self._non_causal_mask = torch.ones(
                # Only take the token length, we expand to fit the batch sizes
                # later.
                (1, seq_len),
                device=self._device,
                dtype=torch.bool,
            )
        return self._non_causal_mask

    def generate(self, non_causal_tokens, causal_tokens, tokenizer, num_new_tokens):
        if len(causal_tokens) == 0:
            full_input = non_causal_tokens
        else:
            full_input = torch.cat([non_causal_tokens, causal_tokens], dim=-1)
        # Causal model.
        gen_tokens = self.llm.generate(full_input, max_new_tokens=num_new_tokens)
        return tokenizer.batch_decode(
            gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )


def setup_peft_module(model, peft_full_att_params, peft_settings):
    # To figure out the possible names:
    # print([(n, type(m)) for n, m in self.vlm.named_modules()])
    if peft_full_att_params:
        target_modules = r"model\.layers\.\d+\.self_attn\.(q_proj|v_proj|k_proj|o_proj)"
    else:
        target_modules = r"model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
    peft_config = LoraConfig(
        target_modules=target_modules,
        modules_to_save=["lm_head"],
        **peft_settings,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model
