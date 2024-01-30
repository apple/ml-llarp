#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import logging
from itertools import chain
from typing import Any, Dict, List

import einops
import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
from habitat import logger
from habitat.tasks.rearrange.multi_task.pddl_action import PddlAction
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.logging import baselines_logger
from habitat_baselines.rl.hrl.hierarchical_policy import HierarchicalPolicy
from habitat_baselines.rl.hrl.hl.fixed_policy import FixedHighLevelPolicy
from habitat_baselines.rl.hrl.hl.high_level_policy import HighLevelPolicy
from habitat_baselines.rl.hrl.skills import NoopSkillPolicy  # noqa: F401.
from habitat_baselines.rl.hrl.skills import ResetArmSkill, WaitSkillPolicy
from habitat_baselines.rl.ppo.policy import CriticHead, PolicyActionData
from habitat_baselines.utils.common import (CategoricalNet,
                                            CustomFixedCategorical)
from habitat_baselines.utils.timing import g_timer

from llarp.dataset.utils import get_category_info
from llarp.policies.cores import *
from llarp.task.utils import get_allowed_actions, get_pddl


def _map_state_dict(state_dict, rename_ckpt_keys):
    return {
        _map_state_dict_k(state_dict_k, rename_ckpt_keys): v
        for state_dict_k, v in state_dict.items()
    }


def _map_state_dict_k(state_dict_k, rename_ckpt_keys):
    for orig_k, replace_k in rename_ckpt_keys.items():
        state_dict_k = state_dict_k.replace(orig_k, replace_k)
    return state_dict_k


@baseline_registry.register_policy
class CatHierarchicalPolicy(HierarchicalPolicy):
    def __init__(
        self,
        config,
        full_config,
        observation_space,
        action_space,
        orig_action_space,
        num_envs,
        aux_loss_config,
        agent_name,
    ):
        super().__init__(
            config,
            full_config,
            observation_space,
            action_space,
            orig_action_space,
            num_envs,
            aux_loss_config,
            agent_name,
        )
        self._strict_loading = (
            config.hierarchical_policy.high_level_policy.strict_loading
        )

        pretrain_path = config.hierarchical_policy.high_level_policy.pretrain_ckpt_path
        if pretrain_path is not None:
            logger.info(f"Loading {pretrain_path}")
            # Load in pre-trained checkpoint.
            ckpt = torch.load(
                pretrain_path,
                map_location="cpu",
            )
            self.load_state_dict(ckpt["state_dict"])

    @property
    def policy_action_space(self):
        return self._high_level_policy.policy_action_space

    @property
    def recurrent_hidden_size(self) -> int:
        if isinstance(self._high_level_policy, FixedHighLevelPolicy):
            # We are using a non-learning based policy
            return 0
        else:
            return self._high_level_policy.recurrent_hidden_size

    def load_state_dict(self, state_dict, **kwargs):
        kwargs["strict"] = self._strict_loading
        state_dict = _map_state_dict(
            state_dict,
            {
                "_high_level_policy._policy_core.vis_bridge.": "_high_level_policy._policy_core.vis_bridge_net.",
                "_high_level_policy._policy_core.action_proj.": "_high_level_policy._policy_core.action_proj_net.",
            },
        )
        return super().load_state_dict(state_dict, **kwargs)

    @property
    def visual_encoder(self):
        return self._high_level_policy.visual_encoder

    def _create_pddl(self, full_config, config):
        all_cats, obj_cats, _ = get_category_info()
        return get_pddl(full_config.habitat.task, all_cats, obj_cats)

    def _get_hl_policy_cls(self, config):
        return eval(config.hierarchical_policy.high_level_policy.name)

    def get_extra(self, action_data, infos, dones) -> List[Dict[str, float]]:
        extra = super().get_extra(action_data, infos, dones)
        # Also add the value information.
        if action_data is not None and action_data.values is not None:
            for i, val_pred in enumerate(action_data.values):
                extra[i]["value"] = val_pred.item()
        return extra

    def _create_skills(self, skills, observation_space, action_space, full_config):
        for action in self._pddl.get_ordered_actions():
            if "pick" in action.name and action.name not in skills:
                # Duplicate the pick skill config.
                skills[action.name] = skills["pick"]
        skill_i = 0
        for (
            skill_name,
            skill_config,
        ) in skills.items():
            cls = eval(skill_config.skill_name)
            skill_policy = cls.from_config(
                skill_config,
                observation_space,
                action_space,
                self._num_envs,
                full_config,
            )
            skill_policy.set_pddl_problem(self._pddl)
            if skill_config.pddl_action_names is None:
                action_names = [skill_name]
            else:
                action_names = skill_config.pddl_action_names
            for skill_id in action_names:
                self._name_to_idx[skill_id] = skill_i
                self._idx_to_name[skill_i] = skill_id
                self._skills[skill_i] = skill_policy
                skill_i += 1


class HlPolicy(HighLevelPolicy):
    def __init__(self, *args, action_space, **kwargs):
        super().__init__(*args, action_space=action_space, **kwargs)
        self._all_actions = get_allowed_actions(
            self._pddl_prob, self._config.allowed_actions
        )
        self._n_actions = len(self._all_actions)
        print(f"Got {self._n_actions} hl actions")

        # Only take the keys that are in the observation space.
        use_obs_space = spaces.Dict(
            {
                k: self._obs_space.spaces[k]
                for k in self._config.policy_input_keys
                if k in self._obs_space.spaces
            }
        )
        # self._prev_sig = None

        self._use_term_action = self._config.use_term_action
        if self._use_term_action:
            self._n_actions += 1

        policy_core_cls = eval(self._config.policy_core_type)
        self._policy_core = policy_core_cls(use_obs_space, action_space, self._config)

        self._policy = CategoricalNet(self._config.ac_hidden_size, self._n_actions)
        self._critic = CriticHead(self._config.ac_hidden_size)

    @property
    def visual_encoder(self):
        return self._policy_core.visual_encoder

    @property
    def recurrent_hidden_size(self):
        return self._policy_core.rnn_hidden_dim

    @property
    def policy_action_space(self):
        return spaces.Discrete(self._n_actions)

    def parameters(self):
        return chain(
            self._policy_core.get_trainable_params(),
            self._policy.parameters(),
            self._critic.parameters(),
        )

    def get_policy_components(self) -> List[nn.Module]:
        return [self._policy_core, self._policy, self._critic]

    def to(self, device):
        self._device = device
        self._critic.to(device)
        self._policy.to(device)
        self._policy_core.to(device)
        return self

    @property
    def should_load_agent_state(self):
        return True

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self._policy_core.forward(observations, rnn_hidden_states, masks)
        return self._critic(features)

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        action,
        rnn_build_seq_info,
    ):
        with g_timer.avg_time(f"hl_policy.eval_actions.get_distrib", level=1):
            distrib, features, _ = self.get_distrib(
                observations, rnn_hidden_states, masks, rnn_build_seq_info
            )
        value = self._critic(features)

        if len(action.shape) == 3:
            # [batch_size, seq_len, data_dim]
            log_probs = distrib.log_prob(einops.rearrange(action, "b t 1 -> b t"))
            log_probs = einops.rearrange(log_probs, "b t -> b t 1")
        else:
            log_probs = distrib.log_probs(action)

        distribution_entropy = distrib.entropy()

        return (
            value,
            log_probs,
            distribution_entropy,
            rnn_hidden_states,
            {},
        )

    def get_distrib(
        self,
        observations,
        rnn_hidden_states,
        masks,
        rnn_build_seq_info=None,
        flatten=False,
    ):
        features, rnn_hidden_states = self._policy_core.forward(
            observations, rnn_hidden_states, masks, rnn_build_seq_info
        )
        distrib = self._policy(features)

        return distrib, features, rnn_hidden_states

    def get_next_skill(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        plan_masks,
        deterministic,
        log_info,
    ):
        next_skill = torch.zeros(self._num_envs, dtype=torch.long)
        skill_args_data: List[Any] = [None for _ in range(self._num_envs)]
        immediate_end = torch.zeros(self._num_envs, dtype=torch.bool)

        with g_timer.avg_time(f"hl_policy.act.get_distrib", level=1):
            distrib, features, rnn_hidden_states = self.get_distrib(
                observations, rnn_hidden_states, masks
            )
        values = self._critic(features)

        if deterministic:
            skill_sel = distrib.mode()
        else:
            skill_sel = distrib.sample()
        action_log_probs = distrib.log_probs(skill_sel)

        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan != 1.0:
                continue
            batch_skill_sel = skill_sel[batch_idx]
            if batch_skill_sel >= len(self._all_actions):
                assert self._use_term_action
                immediate_end[batch_idx] = True
                log_info[batch_idx]["nn_action"] = "terminate"

                # Choose a random skill because it won't matter since we will terminate the episode.
                use_ac = self._all_actions[0]
                next_skill[batch_idx] = self._skill_name_to_idx[use_ac.name]
                skill_args_data[batch_idx] = [
                    entity.name for entity in use_ac.param_values
                ]
            else:
                use_ac = self._all_actions[batch_skill_sel]

                next_skill[batch_idx] = self._skill_name_to_idx[use_ac.name]
                skill_args_data[batch_idx] = [
                    entity.name for entity in use_ac.param_values
                ]
                log_info[batch_idx]["nn_action"] = use_ac.compact_str

        return (
            next_skill,
            skill_args_data,
            immediate_end,
            PolicyActionData(
                action_log_probs=action_log_probs,
                values=values,
                actions=skill_sel,
                rnn_hidden_states=rnn_hidden_states,
            ),
        )

    @property
    def num_recurrent_layers(self):
        return self._policy_core.get_num_rnn_layers()
