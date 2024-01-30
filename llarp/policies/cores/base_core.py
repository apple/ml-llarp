#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import abc

import gym.spaces as spaces
import torch.nn as nn


class PolicyCore(nn.Module, abc.ABC):
    def __init__(self, obs_space, config):
        super().__init__()
        self._im_obs_space = spaces.Dict(
            {
                k: v
                for k, v in obs_space.items()
                if len(v.shape) == 3 and k not in ["third_rgb", "top_down_rgb"]
            }
        )

        self._state_obs_space = spaces.Dict(
            {k: v for k, v in obs_space.items() if len(v.shape) == 1}
        )
        self._config = config
        self._is_blind = len(self._im_obs_space) == 0
        self._prefix_tokens_obs_k = config.prefix_tokens_obs_k

    @property
    @abc.abstractmethod
    def rnn_hidden_dim(self):
        pass

    @property
    def visual_encoder(self):
        return None

    @property
    def hidden_window_dim(self):
        return 512

    @abc.abstractmethod
    def get_num_rnn_layers(self):
        pass

    @abc.abstractmethod
    def get_trainable_params(self):
        pass
