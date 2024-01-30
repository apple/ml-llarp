#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import abc
import copy
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple

import einops
import gym.spaces as spaces
import hydra
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from habitat.core.logging import logger
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet, resnet
from habitat_baselines.rl.ddppo.policy.resnet_policy import ResNetEncoder
from habitat_baselines.rl.models.rnn_state_encoder import \
    build_rnn_state_encoder
from habitat_baselines.utils.timing import g_timer
from torch import Tensor
from torch.nn import functional as F
from transformers import (AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer,
                          LlamaModel, LlamaTokenizer, T5Model)

from llarp.policies.action_decoders import ActionDecoder
from llarp.policies.cores.base_core import PolicyCore
from llarp.policies.transformer_storage import (ATT_MASK_K,
                                                FETCH_BEFORE_COUNTS_K,
                                                HIDDEN_WINDOW_K,
                                                START_ATT_MASK_K,
                                                START_HIDDEN_WINDOW_K)
from llarp.policies.utils import LlmWrapper, soft_update_params
from llarp.policies.vis_bridge import VisBridge
from llarp.policies.visual_encoders import VisualEncoderWrapper
from llarp.task.sensors import WindowDebugSensor
from llarp.task.utils import get_parser


def _tokenize_prompts(
    prompts: Dict[str, str], prompt_suffix: str, tokenizer
) -> Dict[str, Tensor]:
    prompt_tokens = {}
    for prompt_name, prompt in prompts.items():
        # Include the special tokens on the prompt because we want the start token.
        prompt_tokens[prompt_name] = tokenizer.encode(prompt, return_tensors="pt")
    return prompt_tokens, tokenizer.encode(
        prompt_suffix, return_tensors="pt", add_special_tokens=False
    )


def _create_kv_cache(
    context_len, n_heads, embed_dim, n_layers, batch_size, device, dtype
):
    embed_dim_per_head = embed_dim // n_heads
    return tuple(
        [
            tuple(
                [
                    torch.zeros(
                        (batch_size, n_heads, context_len, embed_dim_per_head),
                        device=device,
                        dtype=dtype,
                    )
                    for _ in range(2)
                ]
            )
            for _ in range(n_layers)
        ]
    )


class CleanLLMPolicyCore(PolicyCore):
    def __init__(self, obs_space, action_space, config, device):
        super().__init__(obs_space, config)
        self._train_visual_encoder = config.train_visual_encoder
        self._train_action_decoder = config.train_action_decoder
        self._train_vis_bridge = config.train_vis_bridge

        self._debug_mode = self._config.debug_mode

        # Setup the LLM.
        self._llm: LlmWrapper = hydra.utils.instantiate(
            config.llm_wrapper, device=device
        )
        assert isinstance(self._llm, LlmWrapper)

        self.context_len = self._config.context_len
        self.rollout_take_ratio = self._config.rollout_take_ratio
        self._remove_prompt_pad_toks = config.remove_prompt_pad_toks

        self._is_eval_mode = self._config.is_eval_mode

        self._tokenizer = get_parser(config.tokenizer_id)
        self._prompts, self._prompt_suffix = _tokenize_prompts(
            config.prompts, config.prompt_suffix, self._tokenizer
        )

        # Setup the visual encoder.
        self.vis_encoder_net: VisualEncoderWrapper = hydra.utils.instantiate(
            config.vis_encoder, im_obs_space=self._im_obs_space, llm=self._llm
        )
        assert isinstance(self.vis_encoder_net, VisualEncoderWrapper)
        for param in self.vis_encoder_net.parameters():
            param.requires_grad = self._train_visual_encoder

        # Setup the visual bridge.
        self.vis_bridge_net: VisBridge = hydra.utils.instantiate(
            config.visual_bridge,
            vis_encoder_net=self.vis_encoder_net,
            state_obs_space=self._state_obs_space,
            cfg=config,
            llm=self._llm,
        )
        # For some reason this assert breaks the interactive notebook, but the
        # type seems correct.
        # assert isinstance(self.vis_bridge_net, VisBridge)
        if not self._train_vis_bridge:
            for param in self.vis_bridge_net.parameters():
                param.requires_grad = False
        self._num_visual_tokens = self.vis_bridge.num_tokens

        if isinstance(action_space, spaces.Discrete):
            n_actions = 1
            output_ac_dim = action_space.n
            self._is_cont_action = False
        elif isinstance(action_space, spaces.MultiDiscrete):
            n_actions = len(action_space)
            output_ac_dim = action_space[0].n
            self._is_cont_action = False
        elif isinstance(action_space, spaces.Box):
            n_actions = 1
            output_ac_dim = action_space.shape[0]
            self._is_cont_action = True
        else:
            raise ValueError(f"Unrecognized action space {action_space}")

        if config.use_action_inputs:
            self._n_action_tokens = n_actions
        else:
            self._n_action_tokens = 0

        self.action_decoder_net: ActionDecoder = hydra.utils.instantiate(
            config.action_decoder,
            llm=self._llm,
            input_dim=self._llm.d_model,
            output_dim=output_ac_dim,
            action_space=action_space,
            obs_space=obs_space,
        )
        assert isinstance(self.action_decoder_net, ActionDecoder)
        if not self._train_action_decoder:
            for param in self.action_decoder_net.parameters():
                param.requires_grad = False

        self.display_model_info()
        self._kv_cache = None

    def display_model_info(self):
        disp_s = ("-" * 10) + "\n"
        disp_s += "MODEL INFO\n"

        for name, model in [
            ("Visual Encoder", self.vis_encoder_net),
            ("Visual Bridge", self.vis_bridge_net),
            ("LLM", self._llm.base_model),
            ("Action Decoder", self.action_decoder_net),
        ]:
            total = sum(p.numel() for p in model.parameters())
            num_trainable = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            disp_s += f"    - {name}: {total:,} total, {num_trainable:,} trainable\n"
        disp_s += "-" * 10
        print(disp_s)

    def to(self, device):
        self.action_decoder_net.to(device)
        self.vis_bridge_net.to(device)
        self.vis_encoder_net.to(device)
        self._prompts = {k: v.to(device) for k, v in self._prompts.items()}
        self._prompt_suffix = self._prompt_suffix.to(device)
        self._device = device

        self._llm.to(device)
        return self

    @property
    def visual_encoder(self):
        return self.vis_encoder_net

    @property
    def vis_bridge(self):
        return self.vis_bridge_net

    @property
    def hidden_window_dim(self):
        if self._debug_mode:
            return (
                self._num_visual_tokens + self._n_action_tokens + 2,
                self._llm.d_model,
            )
        else:
            return (self._num_visual_tokens + self._n_action_tokens, self._llm.d_model)

    @property
    def rnn_hidden_dim(self):
        if self._is_eval_mode:
            return self.hidden_window_dim
        else:
            # No hidden dim.
            return 1

    def get_trainable_params(self):
        train_modules = []
        if self._train_vis_bridge:
            train_modules.append(self.vis_bridge_net.parameters())
        if self._train_action_decoder:
            train_modules.append(self.action_decoder_net.parameters())
        if self._train_visual_encoder:
            train_modules.append(self.vis_encoder_net.parameters())
        train_modules.append(self._llm.parameters())

        return chain(*train_modules)

    def get_num_rnn_layers(self):
        return 1

    @g_timer.avg_time("core.forward.eval.get_window", level=1)
    def _get_hidden_window_eval(self, obs, att_masks, actions):
        # Recompute all tokens.
        if self._is_blind:
            visual_features = None
        elif PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in obs:
            visual_features = obs[PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY]
        else:
            with g_timer.avg_time("core.visual_encode_eval", level=1):
                visual_features = self.vis_encoder_net(obs)
        # `visual_features` shape: (#batches, #steps, #tokens, token_dim)
        assert len(visual_features.shape) == 4

        hidden_window = self.vis_bridge(visual_features, obs)
        # `hidden_window` shape: (#batches, #steps, #tokens, token_dim)
        assert len(hidden_window.shape) == 4

        # Get the lang tokens
        lang_tokens = obs[self._prefix_tokens_obs_k]
        if lang_tokens.dim() == 3:
            # Then lang tokens are shape [#batches, #steps, #toks]
            # The instruction at the 1st step is used through rest of episode.
            lang_tokens = lang_tokens[:, 0]

        # Add the action tokens.
        if self._n_action_tokens > 0:
            action_embeds = self.action_decoder_net.embed_action(actions, self._llm)
            assert action_embeds.dim() == 4, f"Got {action_embeds.shape=}"
            hidden_window = torch.cat([hidden_window, action_embeds], dim=2)

        if self._debug_mode:
            debug_info = repeat(
                obs[WindowDebugSensor.uuid],
                "b n s -> b n s d",
                d=hidden_window.shape[-1],
            )
            hidden_window = torch.cat([hidden_window, debug_info], dim=2)

        # Create the attention masks.
        batch_size, num_steps = hidden_window.shape[:2]
        device = hidden_window.device

        # Add the previous window.
        fetch_before_counts_cpu = obs[FETCH_BEFORE_COUNTS_K].cpu().numpy().tolist()
        max_fetch = max(fetch_before_counts_cpu)
        # We can only fetch steps up to the context length. This does not cause
        # a problem because we won't ever have more data in the buffer.
        max_fetch = min(max_fetch, self.context_len - num_steps)
        if max_fetch > 0:
            # Expand the window to account for new tokens we must fetch.
            assert hidden_window.dim() == 4
            assert att_masks.dim() == 3
            # Pad the sequence dim on the right, since we will shift data to the right.
            hidden_window = F.pad(hidden_window, (0, 0, 0, 0, 0, max_fetch), value=0.0)
            att_masks = F.pad(att_masks, (0, 0, 0, max_fetch), value=False)

        is_real_token = None
        if self._debug_mode:
            is_real_token = torch.ones_like(att_masks)

        for batch_idx, fetch_before_count in enumerate(fetch_before_counts_cpu):
            if fetch_before_count == 0:
                continue
            # Shift the tensor to the right to make room for the data from before. The data on the right is anyways bad
            hidden_window[batch_idx] = hidden_window[batch_idx].roll(
                shifts=fetch_before_count, dims=0
            )
            att_masks[batch_idx] = att_masks[batch_idx].roll(
                shifts=fetch_before_count, dims=0
            )
            hidden_window[batch_idx, :fetch_before_count] = obs[START_HIDDEN_WINDOW_K][
                batch_idx, -fetch_before_count:
            ]
            att_masks[batch_idx, :fetch_before_count] = obs[START_ATT_MASK_K][
                batch_idx, -fetch_before_count:
            ]
            # att_masks[batch_idx, :fetch_before_count] = True

            if self._debug_mode:
                is_real_token[batch_idx, :fetch_before_count] = False

        if self._debug_mode:
            check_hidden_window(hidden_window, att_masks)

            # Remove the debug info
            hidden_window = hidden_window[:, :, :-2]

        if hidden_window.shape[1] > self.context_len:
            raise ValueError(f"Hidden window shape is too large {hidden_window.shape=}")

        return hidden_window, lang_tokens, att_masks, is_real_token

    def _post_act(self, actions, hidden_window, obs):
        if self._n_action_tokens > 0:
            embed_action = self.action_decoder_net.embed_action(actions, self._llm)
            hidden_window = torch.cat([hidden_window, embed_action], dim=1)

        if self._debug_mode:
            debug_info = repeat(
                obs[WindowDebugSensor.uuid], "b n -> b n d", d=hidden_window.shape[-1]
            )

            hidden_window = torch.cat([hidden_window, debug_info], dim=1)

        return hidden_window

    @g_timer.avg_time("core.forward.rollout.get_window", level=1)
    def _get_hidden_window_rollout(self, obs):
        if self._is_blind:
            latest_visual_feature = None
        elif (
            not self._train_visual_encoder
            and PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in obs
        ):
            latest_visual_feature = obs[
                PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY
            ]
        else:
            # Compute the most recent visual feature
            with g_timer.avg_time("core.visual_encode_rollout", level=1):
                latest_visual_feature = self.vis_encoder_net(obs)

        # latest_visual_feature: Shape (#batches, #tokens, token_dim)
        assert len(latest_visual_feature.shape) == 3

        with g_timer.avg_time("core.rollout.vis_bridge", level=1):
            latest_hidden_window = self.vis_bridge(latest_visual_feature, obs)
        # Shape (#batches, #tokens, hidden_dim)
        assert len(latest_hidden_window.shape) == 3

        return latest_hidden_window

    @g_timer.avg_time(f"core.update_actions", level=1)
    def update_actions(self, obs, actions):
        att_masks = obs[ATT_MASK_K]
        # Evaluating the policy on a batch of trajectories.
        (
            hidden_toks,
            lang_tokens,
            att_masks,
            is_real_token,
        ) = self._get_hidden_window_eval(obs, att_masks, actions)

        # Flatten the sequences.
        num_toks = hidden_toks.shape[2]
        hidden_out = rearrange(hidden_toks, "b n t d -> b (n t) d")
        att_masks = rearrange(
            repeat(att_masks, "b n 1 -> b n t", t=num_toks),
            "b n t -> b (n t) 1",
        )

        assert (
            lang_tokens.sum() != 0
        ), f"Bad lang tokens: {lang_tokens}, obs {obs['vocab_lang_goal']}"

        n_instruct_seq = lang_tokens.shape[1]
        lang_tokens = self._include_prompt(lang_tokens)

        assert (
            hidden_out.shape[:2] == att_masks.shape[:2]
        ), "Att mask is not proper shape"

        with g_timer.avg_time(f"core.forward.eval.llm_decoder", level=1):
            llm_hidden_out, _ = self._llm.decode(lang_tokens, hidden_out, att_masks)

        # Compute log statistics.
        n_total_seq = hidden_out.shape[1] + lang_tokens.shape[1]
        log_stats = {
            "n_lang_seq": lang_tokens.shape[1],
            "n_obs_seq": hidden_out.shape[1],
            "n_total_seq": n_total_seq,
            "n_batches": hidden_out.shape[0],
            "n_total_toks": hidden_out.shape[0] * n_total_seq,
            "n_prompt_seq": lang_tokens.shape[1] - n_instruct_seq,
            "n_instruct_seq": n_instruct_seq,
            "n_fetch_before": obs[FETCH_BEFORE_COUNTS_K].sum().item(),
        }

        llm_hidden_out = rearrange(
            llm_hidden_out,
            "b (n T) d -> b n T d",
            T=self._num_visual_tokens + self._n_action_tokens,
        )

        if self._num_visual_tokens > 1:
            # Only take the hidden activation starting from the last observation.
            llm_hidden_out = llm_hidden_out[:, :, self._num_visual_tokens - 1 :]

        if self._n_action_tokens > 0:
            # We are predicting the next action.
            llm_hidden_out = llm_hidden_out[:, :, :-1]

        # `llm_hidden_out`: Shape (batch_size, #rollout_steps, #tokens-1, token_dim)
        assert len(llm_hidden_out.shape) == 4
        if llm_hidden_out.shape[1] > actions.shape[1]:
            llm_hidden_out = subsel_llm_hidden_out(
                llm_hidden_out,
                actions.shape[1],
                self._debug_mode,
                obs[FETCH_BEFORE_COUNTS_K].tolist(),
                is_real_token,
                obs[ATT_MASK_K],
            )

        assert (
            llm_hidden_out.shape[:2] == actions.shape[:2]
        ), f"{llm_hidden_out.shape=}, {actions.shape=}"

        use_n_action_tokens = llm_hidden_out.shape[2]

        llm_hidden_out = rearrange(llm_hidden_out, "b n t d -> b (n t) d")
        with g_timer.avg_time(f"core.forward.eval.action_decoder_net", level=1):
            vis_tokens = hidden_toks[:, :, -use_n_action_tokens:]
            logits, extra, features = self.action_decoder_net(
                llm_hidden_out, obs, vis_tokens
            )

        logits = rearrange(logits, "b (n t) d -> b n t d", t=use_n_action_tokens)
        features = rearrange(features, "b (n t) d -> b n t d", t=use_n_action_tokens)

        # Take before the action prediction as the hidden state for VF.
        features = features[:, :, 0]

        distrib = self.action_decoder_net.get_distrib(logits, extra=extra)

        log_probs = distrib.log_probs(actions)
        entropy = distrib.entropy()

        # `log_probs` should be shape (#batches, #steps, action_dim)
        assert len(log_probs.shape) == 3
        log_probs = log_probs.mean(-1, keepdims=True)

        if entropy.dim() == 4:
            # Don't keep dims because the FixedCategorical entropy pads a dim.
            entropy = entropy.mean(-2)
        # `entropy` should be (#batches, #steps, 1)
        assert entropy.dim() == 3

        # Logits should correspond to actions.
        return log_probs, features, entropy, log_stats

    def _include_prompt(self, lang_tokens: Tensor) -> Tuple[Tensor, int]:
        """
        Operates over language token ids.
        """

        all_toks = []
        batch_size = lang_tokens.shape[0]

        use_prompt = self._prompts["habitat"]
        if use_prompt.shape[1] > 1:
            all_toks.append(repeat(use_prompt, "1 s -> b s", b=batch_size))
        all_toks.append(lang_tokens)

        if self._prompt_suffix.shape[1] > 0:
            all_toks.append(repeat(self._prompt_suffix, "1 s -> b s", b=batch_size))

        return torch.cat(all_toks, dim=-1)

    @property
    def n_toks_per_step(self):
        return self.vis_bridge_net.num_tokens + self._n_action_tokens

    def _construct_kv_cache(self, obs, masks):
        n_obs_toks = self.vis_bridge_net.num_tokens
        use_ctx_len = self.n_toks_per_step * self.context_len

        self._n_lang_toks = 0
        self._n_lang_toks += self._prompt_suffix.shape[1]
        if self._prompts["habitat"].shape[1] > 1:
            self._n_lang_toks += self._prompts["habitat"].shape[1]
        self._n_lang_toks += obs[self._prefix_tokens_obs_k].shape[1]

        use_ctx_len += self._n_lang_toks

        batch_size = masks.shape[0]
        device = masks.device
        logger.info(f"Creating KV cache of length {use_ctx_len}")

        cfg = self._llm.config
        # Some huggingface LLMs have different naming for the num kv heads.
        if hasattr(cfg, "num_key_value_heads"):
            num_kv_heads = cfg.num_key_value_heads
        else:
            num_kv_heads = cfg.num_kv_attention_heads

        self._kv_cache = _create_kv_cache(
            # Since we don't include the current observation
            context_len=use_ctx_len - 1,
            n_heads=num_kv_heads,
            embed_dim=cfg.hidden_size,
            n_layers=cfg.num_hidden_layers,
            batch_size=batch_size,
            device=device,
            dtype=torch.bfloat16,
        )

        self._kv_cache_masks = torch.zeros(
            (batch_size, use_ctx_len), dtype=torch.bool, device=device
        )

        self._kv_counts = torch.zeros(
            (batch_size, n_obs_toks),
            dtype=torch.long,
            device=device,
        )

    @g_timer.avg_time(f"core.rollout_actions", level=1)
    def rollout_actions(self, obs, masks, deterministic=False, only_features=False):
        if self._kv_cache is None:
            self._construct_kv_cache(obs, masks)

        embed_obs = self._get_hidden_window_rollout(obs)
        lang_toks = obs[self._prefix_tokens_obs_k]

        lang_toks = self._include_prompt(lang_toks)

        # This will reset the episode if we are past the sequence length.
        max_seq_len = self._kv_cache_masks.shape[1]
        use_masks = masks * (self._kv_counts.max(1, keepdims=True).values < max_seq_len)

        # Where the environment reset, we must recompute the kv cache for the
        # instructions.
        (
            self._kv_cache,
            self._kv_cache_masks,
            self._kv_counts,
        ) = _construct_lang_tok_kv_cache(
            self._llm,
            lang_toks,
            self._kv_cache,
            self._kv_cache_masks,
            self._kv_counts,
            use_masks,
            self.vis_bridge_net.num_tokens,
            self._remove_prompt_pad_toks,
        )

        (
            actions,
            log_probs,
            features,
            self._kv_cache,
            self._kv_cache_masks,
            self._kv_counts,
        ) = _auto_regr_sample_actions_kv_cache(
            self._n_action_tokens,
            self._llm,
            self.action_decoder_net,
            only_features,
            deterministic,
            self._kv_cache,
            self._kv_cache_masks,
            self._kv_counts,
            embed_obs,
            self._n_lang_toks,
            obs,
        )
        if not only_features:
            embed_obs = self._post_act(actions, embed_obs, obs)

        return actions, log_probs, features, embed_obs


@g_timer.avg_time("advance_cache", level=1)
def _advance_cache(
    kv_cache,
    kv_cache_masks,
    kv_counts,
    n_cur_toks,
    n_next_toks,
    new_kv_cache,
    prompt_len,
):
    kv_counts += n_cur_toks
    max_seq_len = kv_cache_masks.shape[1]
    # We will write past on the last step.
    exceeds_context = (kv_counts >= (max_seq_len + n_cur_toks)).any(-1)

    if exceeds_context.sum() > 0:
        raise ValueError(
            f"The episode length exceeds the max supported context length. Streaming LLM is currently not supported. {kv_counts=}, {max_seq_len=}"
        )

    kv_cache = tuple(
        [
            tuple([new_kv_cache[i][j][:, :, 1:] for j in range(len(kv_cache[i]))])
            for i in range(len(kv_cache))
        ]
    )

    # Push cache to left to make room for next incoming tokens.
    kv_cache_masks = kv_cache_masks.roll(shifts=-n_next_toks, dims=1)
    # We want to attend to the next observations
    kv_cache_masks[:, -n_next_toks:] = True

    return kv_cache, kv_cache_masks, kv_counts


@g_timer.avg_time("write_kv_cache", level=1)
def _write_to_cache(dst_kv_cache, src_kv_cache, write_batch_idxs):
    """
    Writes to the end of the seq dim.
    """
    assert len(dst_kv_cache) == len(src_kv_cache)

    for i in range(len(dst_kv_cache)):
        for j in range(len(dst_kv_cache[i])):
            # Clear contents of KV cache.
            dst_kv_cache[i][j][write_batch_idxs] *= 0.0
            # Right of the tensor are the more recent tokens.
            dst_kv_cache[i][j][
                write_batch_idxs, :, -src_kv_cache[i][j].shape[2] :
            ] = src_kv_cache[i][j]


@g_timer.avg_time("construct_kv_cache", level=1)
def _construct_lang_tok_kv_cache(
    llm,
    lang_tokens,
    kv_cache,
    kv_cache_masks: Tensor,
    kv_counts: Tensor,
    masks: Tensor,
    n_obs_toks: int,
    remove_prompt_pad_toks: bool,
):
    """
    :returns: Updated kv cache, kv masks, and kv counts.
    """

    batch_size, n_lang_toks = lang_tokens.shape
    n_cache_toks = kv_cache[0][0].shape[2]
    device = lang_tokens.device

    write_batch_idxs = rearrange(~masks, "b 1 -> b")
    if write_batch_idxs.sum() != 0:
        # New instruction prompts.
        # Update the attention masks over the KV cache.
        # Disable attention to old data.
        kv_cache_masks[write_batch_idxs] = False
        # Most recent obs should also have attention
        kv_cache_masks[write_batch_idxs, -(n_lang_toks + n_obs_toks) :] = True

        if remove_prompt_pad_toks:
            # Don't include attention to pad tokens in the prompt. Note that
            # these pad tokens will occur in the middle of the prompt.
            kv_cache_masks[
                write_batch_idxs, -(n_lang_toks + n_obs_toks) : -n_obs_toks
            ] *= (lang_tokens[write_batch_idxs] != 0)

        # Reset the token position IDs.
        kv_counts[write_batch_idxs] = torch.arange(
            start=n_lang_toks,
            end=n_lang_toks + n_obs_toks,
            dtype=kv_counts.dtype,
            device=kv_counts.device,
        )

        lang_kv_cache = llm.base_model.forward(
            input_ids=lang_tokens[write_batch_idxs],
            attention_mask=kv_cache_masks[write_batch_idxs, -n_lang_toks:].int(),
            use_cache=True,
        ).past_key_values

        _write_to_cache(kv_cache, lang_kv_cache, write_batch_idxs)

    return kv_cache, kv_cache_masks, kv_counts


@g_timer.avg_time("core.rollout.auto_regr_sample", level=1)
def _auto_regr_sample_actions_kv_cache(
    n_action_toks: int,
    llm: LlmWrapper,
    action_decoder_net: ActionDecoder,
    only_features: bool,
    deterministic: bool,
    kv_cache,
    kv_cache_masks,
    kv_counts,
    embed_obs: Tensor,
    prompt_len: int,
    obs,
):
    actions = []
    log_probs = []
    features = None

    n_obs_toks = embed_obs.shape[1]
    n_action_preds = max(1, n_action_toks)
    for action_idx in range(n_action_preds):
        n_toks_this_step = embed_obs.shape[1]
        if n_toks_this_step - 1 > 0:
            sub_sel_kv_cache = _extract_from_kv_cache(kv_cache, n_toks_this_step - 1)
        else:
            sub_sel_kv_cache = kv_cache

        seq_out = llm.base_model.forward(
            inputs_embeds=embed_obs.to(torch.bfloat16),
            attention_mask=kv_cache_masks.int(),
            use_cache=True,
            position_ids=kv_counts[:, :n_toks_this_step],
            past_key_values=sub_sel_kv_cache,
        )

        llm_hidden_out = seq_out.last_hidden_state[:, -1].to(torch.float32)
        new_kv_cache = seq_out.past_key_values
        assert len(llm_hidden_out.shape) == 2

        with g_timer.avg_time(f"core.forward.rollout.action_decoder_net", level=1):
            logits, extra, last_features = action_decoder_net(
                llm_hidden_out, obs, embed_obs
            )

        if features is None:
            features = last_features

        if only_features:
            return None, None, features, kv_cache, kv_cache_masks, kv_counts

        distrib = action_decoder_net.get_distrib(logits, extra=extra)
        if deterministic:
            actions_i = distrib.mode()
        else:
            actions_i = distrib.sample()

        log_probs.append(distrib.log_probs(actions_i))
        actions.append(actions_i)

        if (action_idx + 1) < n_action_preds:
            # We will generate an action next.
            embed_obs = action_decoder_net.embed_action(actions_i, llm)
            kv_cache, kv_cache_masks, kv_counts = _advance_cache(
                kv_cache,
                kv_cache_masks,
                kv_counts,
                n_cur_toks=n_toks_this_step,
                # Actions are inputs.
                n_next_toks=1,
                new_kv_cache=new_kv_cache,
                prompt_len=prompt_len,
            )

    if n_action_toks > 1:
        # We must include the final action prediction in the output as well.
        seq_out = llm.base_model.forward(
            inputs_embeds=embed_obs.to(torch.bfloat16),
            attention_mask=kv_cache_masks.int(),
            use_cache=True,
            position_ids=kv_counts[:, :1],
            past_key_values=kv_cache,
        )
        new_kv_cache = seq_out.past_key_values

    kv_cache, kv_cache_masks, kv_counts = _advance_cache(
        kv_cache,
        kv_cache_masks,
        kv_counts,
        n_cur_toks=n_toks_this_step,
        # Always start with a batch of observation tokens next.
        n_next_toks=n_obs_toks,
        new_kv_cache=new_kv_cache,
        prompt_len=prompt_len,
    )

    actions = torch.cat(actions, dim=-1)
    # `actions`: Shape (batch_size, action_dim)
    assert len(actions.shape) == 2

    log_probs = torch.cat(log_probs, dim=-1).mean(-1, keepdims=True)
    # `log_probs`: Shape (batch_size, 1)
    assert len(log_probs.shape) == 2
    return actions, log_probs, features, kv_cache, kv_cache_masks, kv_counts


def _extract_from_kv_cache(kv_cache, remove_toks):
    """
    Extracts from the RIGHT SIDE (left side is chopped off).
    """

    new_kv_cache = []
    for cache in kv_cache:
        new_cache = []
        for layer_cache in cache:
            new_cache.append(layer_cache[:, :, remove_toks:])
        new_kv_cache.append(tuple(new_cache))

    return tuple(new_kv_cache)


def check_hidden_window(hidden_window, att_masks):
    debug_info = get_debug(hidden_window)

    # The sequence should always be valid at the left and possibly invalid at
    # the right since we will insert the prompt tokens to the left.
    if not att_masks[:, 0].all():
        raise ValueError(f"Bad attention masks start {att_masks[:, 0]}")

    for batch_idx in range(debug_info.shape[0]):
        batch_step_info = debug_info[batch_idx, :, 0][att_masks[batch_idx].view(-1)]
        batch_ep_info = debug_info[batch_idx, :, 1][att_masks[batch_idx].view(-1)]

        # Each step should be increasing by 1.
        if not ((batch_step_info[:-1] + 1) == batch_step_info[1:]).all():
            raise ValueError(
                f"Episode steps at batch {batch_idx} not incrementing {batch_step_info}"
            )

        # Steps should belong to the same episode.
        if not (batch_ep_info[0] == batch_ep_info).all():
            raise ValueError(f"Episode IDs at at batch {batch_idx} not consistent")


def get_debug(hidden_window):
    return hidden_window[..., -2:, 0].int()


def subsel_llm_hidden_out(
    llm_hidden_out, real_n_steps, debug_mode, fetch_befores, is_real_token, att_masks
):
    # We padded dimensions to fetch steps from the previous rollout.
    # These steps aren't used in the loss.
    if debug_mode:
        all_is_real_token = torch.stack(
            [
                is_real_token[batch_idx].roll(shifts=-fetch_before, dims=0)[
                    :real_n_steps
                ]
                for batch_idx, fetch_before in enumerate(fetch_befores)
            ],
            dim=0,
        )

        # It's okay if it's not a real token where the attention mask is False.
        if not torch.logical_or(all_is_real_token, ~att_masks).all():
            raise ValueError("Matching bad tokens")

    # Undo the effect of the roll and remove the fetch before inputs.
    return torch.stack(
        [
            llm_hidden_out[batch_idx].roll(shifts=-fetch_before, dims=0)[:real_n_steps]
            for batch_idx, fetch_before in enumerate(fetch_befores)
        ],
        dim=0,
    )
