#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import abc

import hydra
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from flamingo_pytorch import PerceiverResampler


class VisBridge(abc.ABC, nn.Module):
    @abc.abstractmethod
    def forward(self, vis_features, obs):
        pass

    @property
    @abc.abstractmethod
    def num_tokens(self) -> int:
        pass


class ResamplerVisBridge(VisBridge):
    def __init__(
        self,
        vis_encoder_net,
        llm,
        resampler_depth,
        resampler_dim_head,
        resampler_heads,
        num_output_latents,
        use_b16: bool,
        **kwargs,
    ):
        super().__init__()
        if use_b16:
            self._create_type = torch.bfloat16
        else:
            self._create_type = torch.float32

        num_visual_tokens, vis_token_dim = vis_encoder_net.output_shape

        self.token_resampler = PerceiverResampler(
            dim=vis_token_dim,
            num_media_embeds=num_visual_tokens,
            num_latents=num_output_latents,
            depth=resampler_depth,
            dim_head=resampler_dim_head,
            heads=resampler_heads,
        )
        self.up_proj = nn.Linear(vis_token_dim, llm.d_model)
        self.token_resampler.to(self._create_type)
        self._num_output_tokens = num_output_latents

    def forward(self, vis_features, obs):
        """
        Always returns float32 data type regardless of net internal type.
        """

        if len(vis_features.shape) == 4:
            orig_batch_size = vis_features.shape[0]
            vis_features = rearrange(vis_features, "b n t d -> (b n) t d")
        else:
            orig_batch_size = None

        vis_features = vis_features.to(self._create_type)
        embeds = self.token_resampler(vis_features)
        # The token resampler outputs another dimension for some reason...
        embeds = rearrange(embeds, "b 1 t d -> b t d")
        embeds = self.up_proj(embeds)

        if orig_batch_size is not None:
            embeds = rearrange(embeds, "(b n) t d -> b n t d", b=orig_batch_size)
        return embeds.to(torch.float32)

    @property
    def num_tokens(self) -> int:
        return self._num_output_tokens


class MlpVisBridge(VisBridge):
    """
    For operating over single token inputs.
    """

    def __init__(
        self,
        vis_encoder_net,
        llm,
        state_obs_space,
        hidden_size,
        cfg,
        **kwargs,
    ):
        super().__init__()
        llm_input_size = llm.d_model
        if not hasattr(vis_encoder_net, "embd_size"):
            if hasattr(vis_encoder_net, "output_shape"):
                input_size = np.prod(vis_encoder_net.output_shape)
            else:
                raise ValueError("Visual encoder must specify output size.")
        else:
            input_size = vis_encoder_net.embd_size

        self.visual_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                input_size,
                hidden_size,
            ),
            nn.ReLU(True),
        )
        self._state_obs_space = state_obs_space
        visual_dim = hidden_size
        input_dim = visual_dim + sum(
            space.shape[0] for space in state_obs_space.spaces.values()
        )
        self.state_token_proj = nn.Linear(input_dim, llm_input_size)

    def forward(self, vis_features, obs):
        # There is only 1 visual token. Extract this token and expand.

        if len(vis_features.shape) == 4:
            # Operate on the only visual token.
            assert vis_features.shape[2] == 1

            batch_size = vis_features.shape[0]
            # Flatten and remove #token dim.
            vis_features = rearrange(vis_features, "b r 1 d -> (b r) d")
            vis_features = self.visual_fc(vis_features)
            vis_features = rearrange(vis_features, "(b r) d -> b r d", b=batch_size)
        else:
            assert vis_features.shape[1] == 1
            vis_features = vis_features[:, 0]

            vis_features = self.visual_fc(vis_features)

        state_features = [obs[k] for k in self._state_obs_space.keys()]

        if vis_features is None:
            hidden_window = torch.cat(state_features, dim=-1)
        elif len(state_features) == 0:
            hidden_window = vis_features
        else:
            hidden_window = torch.cat([vis_features, *state_features], dim=-1)

        hidden_window = self.state_token_proj(hidden_window)
        return hidden_window.unsqueeze(-2)

    @property
    def num_tokens(self) -> int:
        return 1
