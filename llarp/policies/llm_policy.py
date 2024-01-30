#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import logging
from collections import defaultdict
from itertools import chain
from typing import Any, Dict, List

import gym.spaces as spaces
import hydra
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from habitat import logger
from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hierarchical_policy import HierarchicalPolicy
from habitat_baselines.rl.hrl.hl.fixed_policy import FixedHighLevelPolicy
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.hrl.skills import NoopSkillPolicy  # noqa: F401.
from habitat_baselines.rl.hrl.skills import ResetArmSkill, WaitSkillPolicy
from habitat_baselines.rl.ppo.policy import (CriticHead, Policy,
                                             PolicyActionData)
from habitat_baselines.rl.ppo.ppo_trainer import get_device
from habitat_baselines.utils.timing import g_timer

from llarp.dataset.utils import get_category_info
from llarp.policies.cores import *
from llarp.task.utils import get_allowed_actions, get_pddl

EPS_PPO = 1e-5


@baseline_registry.register_policy
class LlmPolicy(nn.Module, Policy):
    def __init__(self, config, obs_space, action_space, **kwargs):
        Policy.__init__(self, action_space)
        nn.Module.__init__(self)
        self._obs_space = obs_space

        policy_cfg = (
            config.habitat_baselines.rl.policy.main_agent.hierarchical_policy.high_level_policy
        )
        device = get_device(config)
        # Only take the keys that are in the observation space.
        use_obs_space = spaces.Dict(
            {
                k: self._obs_space.spaces[k]
                for k in policy_cfg.policy_input_keys
                if k in self._obs_space.spaces
            }
        )

        self._use_term_action = policy_cfg.use_term_action

        policy_core_cls = eval(policy_cfg.policy_core_type)
        self._policy_core = policy_core_cls(
            use_obs_space, action_space, policy_cfg, device
        )

        self._critic = hydra.utils.instantiate(
            policy_cfg.critic,
            input_size=self._policy_core.action_decoder_net.hidden_size,
            n_envs=config.habitat_baselines.num_environments,
            device=device,
            obs_space=self._obs_space,
        )
        self._resume_path = config.habitat_baselines.resume_ckpt_path

    @property
    def policy_core(self):
        return self._policy_core

    @property
    def visual_encoder(self):
        return self._policy_core.visual_encoder

    @property
    def recurrent_hidden_size(self):
        return self._policy_core.rnn_hidden_dim

    def parameters(self):
        return chain(
            self._policy_core.get_trainable_params(),
            self._critic.parameters(),
        )

    def _get_policy_components(self) -> List[nn.Module]:
        return [self._policy_core, self._critic]

    def to(self, device):
        self._device = device
        self._critic.to(device)
        self._policy_core.to(device)
        if self._resume_path is not None:
            ckpt = torch.load(self._resume_path, map_location="cpu")
            self.load_state_dict(ckpt["state_dict"])
            logger.info(f"Loaded from step {ckpt['extra_state']['step']:,}")
            self._resume_path = None
        return self

    @property
    def should_load_agent_state(self):
        return True

    def get_value(self, observations, hidden_states, prev_actions, masks):
        (
            _,
            _,
            features,
            _,
        ) = self._policy_core.rollout_actions(observations, masks, only_features=True)
        return self._critic.forward(features, observations)

    def evaluate_actions(
        self,
        observations,
        action,
    ):
        (
            log_probs,
            features,
            distribution_entropy,
            self.policy_core_log,
        ) = self._policy_core.update_actions(observations, action)
        value = self._critic.forward_norm(features, observations)

        return (
            value,
            log_probs,
            distribution_entropy,
            {},
        )

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        (
            actions,
            log_probs,
            features,
            hidden_states,
        ) = self._policy_core.rollout_actions(observations, masks, deterministic)
        values = self._critic.forward(features, observations)

        policy_info = None
        if self._policy_core._is_eval_mode:
            sel_acs = [self._policy_core._tokenizer.decode(ac) for ac in actions]
            policy_info = [{"pred_ac": sel_ac} for sel_ac in sel_acs]

        return PolicyActionData(
            action_log_probs=log_probs,
            values=values,
            actions=actions,
            rnn_hidden_states=hidden_states,
            policy_info=policy_info,
        )

    @property
    def num_recurrent_layers(self):
        return self._policy_core.get_num_rnn_layers()

    @classmethod
    def from_config(cls, config, observation_space, action_space, **kwargs):
        return cls(config, observation_space, action_space, **kwargs)


class LinearCriticHead(nn.Module):
    def __init__(self, input_size, use_b16: bool, **kwargs):
        super().__init__()

        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        if use_b16:
            self._use_type = torch.bfloat16
            self.to(torch.bfloat16)
        else:
            self._use_type = torch.float32

    def post_process_returns(self, returns, obs):
        return returns

    def forward_norm(self, x, obs):
        return self.forward(x, obs)

    def update_stats(self, returns, obs):
        pass

    def forward(self, x, obs):
        return self.fc(x.to(self._use_type))


class DeepCriticHead(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        n_envs: int,
        device,
        obs_space: spaces.Dict,
        use_b16: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(True),
        )
        self.fc = nn.Linear(hidden_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        if use_b16:
            self._use_type = torch.bfloat16
            self.to(torch.bfloat16)
        else:
            self._use_type = torch.float32

    def update_stats(self, returns, obs):
        return

    def post_process_returns(self, returns, obs):
        return returns

    def forward(self, x, obs):
        return self.forward_norm(x, obs)

    def forward_norm(self, x, obs):
        return self.fc(self.proj(x.to(self._use_type)))
