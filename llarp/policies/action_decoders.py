#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import abc
from typing import Optional, Tuple

import gym.spaces as spaces
import torch
import torch.nn as nn
from einops import rearrange
from habitat_baselines.utils.common import CustomFixedCategorical, CustomNormal
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler

from llarp.policies.utils import LlmWrapper

MAX_N_ACTIONS = 10


class FixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        # return super().log_prob(actions)
        if actions.dim() == 2:
            return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1)
        elif actions.dim() == 3:
            return super().log_prob(actions)
        else:
            raise ValueError(f"Unrecognized actions shape {actions.shape}")

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

    def entropy(self):
        return super().entropy().unsqueeze(-1)


class ActionDecoder(nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(
        self, hidden_state, obs, embed_obs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        pass

    @property
    @abc.abstractmethod
    def hidden_size(self) -> int:
        pass

    @abc.abstractmethod
    def embed_action(self, action, llm):
        pass

    def get_distrib(self, logits, **kwargs):
        return FixedCategorical(logits=logits.float(), validate_args=False)


class MlpDecoder(ActionDecoder):
    def __init__(
        self,
        *args,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        action_space,
        use_b16: bool = False,
        min_log_std: int = -5,
        max_log_std: int = 2,
        log_std_init: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(True),
        )
        self._is_cont_action = isinstance(action_space, spaces.Box)

        self.linear = nn.Linear(hidden_size, output_dim)
        self._hidden_size = hidden_size

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

        self._min_log_std = min_log_std
        self._max_log_std = max_log_std
        if self._is_cont_action:
            self.log_std = torch.nn.parameter.Parameter(
                torch.randn(output_dim) * 0.01 + log_std_init
            )
            # Project to embedding of continuous action.
            self.action_embed = nn.Linear(output_dim, input_dim)
        else:
            # Embedding for each option.
            self.action_embed = nn.Embedding(
                num_embeddings=output_dim, embedding_dim=input_dim
            )

        if use_b16:
            self.to(torch.bfloat16)

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    def forward(self, hidden_state, obs, embed_obs):
        hidden_state = self.proj(hidden_state)
        return self.linear(hidden_state), None, hidden_state

    def embed_action(self, action, llm):
        if self._is_cont_action:
            # Add sequence dim
            return self.action_embed(action.unsqueeze(-2))
        else:
            return self.action_embed(action)

    def get_distrib(self, logits, **kwargs):
        if self._is_cont_action:
            if logits.dim() == 4:
                logits = rearrange(logits, "b n 1 d -> b n d")
            log_std = self.log_std
            log_std = torch.clamp(log_std, self._min_log_std, self._max_log_std)
            std = torch.exp(log_std)
            return CustomNormal(logits.float(), std.float(), validate_args=False)
        else:
            return FixedCategorical(logits=logits.float(), validate_args=False)
