#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import random
from collections import defaultdict

import gym.spaces as spaces
import numpy as np
import torch
from habitat_baselines.common.env_factory import VectorEnvFactory
from habitat_baselines.common.tensor_dict import TensorDict


class TestBatchedEnv:
    def __init__(self, n_envs, n_max_steps, prob_end, rgb_name, lang_name):
        # self._im_shape = (226, 226, 3)
        self._im_shape = (336, 336, 3)
        self._vocab_shape = (30,)
        self._n_max_steps = n_max_steps
        self._prob_end = prob_end
        self._rgb_name = rgb_name
        self._lang_name = lang_name
        self.observation_space = spaces.Dict(
            {
                rgb_name: spaces.Box(
                    shape=self._im_shape, dtype=np.uint8, low=0, high=255
                ),
                "debug_info": spaces.Box(shape=(2,), dtype=int, low=0, high=100000000),
                lang_name: spaces.Box(
                    shape=self._vocab_shape,
                    low=0,
                    high=1,
                    dtype=np.int64,
                ),
            }
        )

        self.action_space = spaces.Discrete(70)

        self.observation_spaces = [self.observation_space for _ in range(n_envs)]
        self.action_spaces = [self.action_space for _ in range(n_envs)]
        self.orig_action_spaces = self.action_spaces
        self._n_envs = n_envs
        self._n_steps = defaultdict(lambda: 1)
        self._episode_rollout_id = defaultdict(lambda: random.randint(0, 9999999))

        self.num_envs = self._n_envs
        self._obs = {}
        self._cur_goal = {}

    def reset(self):
        self.num_steps = 1

        ret = [
            {
                self._rgb_name: torch.randint(
                    low=0, high=255, size=self._im_shape, dtype=torch.uint8
                ),
                "debug_info": np.array(
                    [self._n_steps[env_i], self._episode_rollout_id[env_i]],
                    dtype=np.float32,
                ),
                self._lang_name: torch.randint(
                    low=0,
                    high=5_000,
                    size=self._vocab_shape,
                    dtype=torch.int,
                ),
            }
            for env_i in range(self._n_envs)
        ]

        for i in range(self._n_envs):
            self._cur_goal[i] = torch.randint(
                low=0,
                high=5_000,
                size=self._vocab_shape,
                dtype=torch.int,
            )
            ret[i][self._lang_name] = self._cur_goal[i]

        return ret

    def post_step(self, obs):
        return obs

    def async_step_at(self, index_env, action):
        self._n_steps[index_env] += 1
        self._obs[index_env] = {
            self._rgb_name: torch.randint(
                low=0, high=255, size=self._im_shape, dtype=torch.uint8
            )
            + action,
        }

    def wait_step_at(self, index_env):
        done = torch.rand(1).item() < self._prob_end
        if self._n_steps[index_env] >= self._n_max_steps:
            done = True
        info = {}
        reward = torch.randn(1).item()
        if done:
            self._n_steps[index_env] = 1
            self._episode_rollout_id[index_env] = random.randint(0, 9999999)
            self._cur_goal[index_env] = torch.randint(
                low=0,
                high=5_000,
                size=self._vocab_shape,
                dtype=torch.int,
            )
        self._obs[index_env][self._lang_name] = self._cur_goal[index_env]

        self._obs[index_env]["debug_info"] = np.array(
            [self._n_steps[index_env], self._episode_rollout_id[index_env]],
            dtype=np.float32,
        )

        return self._obs[index_env], reward, done, info

    def close(self):
        return


class TestVectorEnvFactory(VectorEnvFactory):
    def construct_envs(
        self,
        config,
        workers_ignore_signals: bool = False,
        enforce_scenes_greater_eq_environments: bool = False,
        is_first_rank: bool = True,
    ):
        return TestBatchedEnv(
            16,
            config.habitat_baselines.test_env_n_max_env_steps,
            config.habitat_baselines.test_env_prob_end,
            config.habitat_baselines.test_env_rgb_name,
            config.habitat_baselines.test_env_lang_name,
        )
