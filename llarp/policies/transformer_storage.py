#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import warnings
from typing import Any, Dict, Iterator, Optional

import gym.spaces as spaces
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.storage import Storage
from habitat_baselines.common.tensor_dict import DictTree, TensorDict
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_pack_info_from_dones, build_rnn_build_seq_info)
from habitat_baselines.utils.common import get_action_space_info
from habitat_baselines.utils.timing import g_timer
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

ATT_MASK_K = "att_mask"
HIDDEN_WINDOW_K = "hidden_window"
START_HIDDEN_WINDOW_K = "start_hidden_window"
START_ATT_MASK_K = "start_att_mask"
FETCH_BEFORE_COUNTS_K = "fetch_before_counts"


def transpose_stack_pad_dicts(dicts_i):
    res = {}
    for k in dicts_i[0].keys():
        if isinstance(dicts_i[0][k], dict):
            res[k] = transpose_stack_pad_dicts([d[k] for d in dicts_i])
        else:
            res[k] = pad_sequence(
                [d[k] for d in dicts_i], batch_first=True, padding_value=0.0
            )

    return res


@baseline_registry.register_storage
class TransformerRolloutStorage(RolloutStorage):
    def __init__(
        self,
        *args,
        numsteps,
        num_envs,
        observation_space,
        actor_critic,
        **kwargs,
    ):
        self._frozen_visual = (
            PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY in observation_space.spaces
        )
        if self._frozen_visual:
            # Remove the head RGB camera because we have the visual information in the visual features
            observation_space = spaces.Dict(
                {k: v for k, v in observation_space.spaces.items() if k != "head_rgb"}
            )

        super().__init__(
            *args,
            numsteps=numsteps,
            num_envs=num_envs,
            observation_space=observation_space,
            actor_critic=actor_critic,
            **kwargs,
        )

        core = actor_critic.policy_core

        self._context_len = core.context_len
        self._debug_mode = core._debug_mode
        self._rollout_take_ratio = core.rollout_take_ratio

        self.hidden_window = torch.zeros(
            num_envs,
            numsteps + self._context_len,
            *core.hidden_window_dim,
        )
        self.att_masks = torch.zeros(num_envs, self._context_len, dtype=torch.bool)

        # The att masks BEFORE this rollout.
        self.before_start_att_masks = torch.zeros(
            num_envs, self._context_len, dtype=torch.bool
        )

    def to(self, device):
        super().to(device)
        self.hidden_window = self.hidden_window.to(device)
        self.att_masks = self.att_masks.to(device)

    @g_timer.avg_time("rollout_storage.insert", level=1)
    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
        **kwargs,
    ):
        if self._frozen_visual and next_observations is not None:
            next_observations = {
                k: v for k, v in next_observations.items() if k != "head_rgb"
            }
        super().insert(
            next_observations=next_observations,
            next_recurrent_hidden_states=None,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards,
            next_masks=next_masks,
            buffer_index=buffer_index,
            **kwargs,
        )

        if next_masks is not None:
            step = self.current_rollout_step_idxs[buffer_index]
            # Shift the tensor over to the left 1.
            self.att_masks = self.att_masks.roll(shifts=-1, dims=1)
            # Start over if we are at a new episode.
            self.att_masks *= next_masks.to(self.device)
            self.att_masks[:, -1] = True

        if next_recurrent_hidden_states is not None:
            write_step = (
                self.current_rollout_step_idxs[buffer_index] + self._context_len - 1
            )
            env_slice = slice(
                int(buffer_index * self._num_envs / self._nbuffers),
                int((buffer_index + 1) * self._num_envs / self._nbuffers),
            )

            self.hidden_window[env_slice, write_step] = next_recurrent_hidden_states

    def insert_first_observations(self, batch):
        if self._frozen_visual:
            batch = {k: v for k, v in batch.items() if k != "head_rgb"}
        super().insert_first_observations(batch)
        self.att_masks[:, -1] = True

    def after_update(self):
        """
        Copy over information from the end of the current rollout buffer into the rollout buffer start.
        """
        super().after_update()
        self.hidden_window[:, : self._context_len] = (
            self.hidden_window[:, -self._context_len :].detach().clone()
        )
        self.before_start_att_masks.copy_(self.att_masks)

        # Clear the rest to write the next rollout
        # self.hidden_window[:, -(self.num_steps - 1) :] = 0.0
        self.hidden_window[:, self._context_len - 1 :] = 0.0

        self.hidden_window[:, : self._context_len - 1] *= rearrange(
            self.att_masks[:, :-1], "b n -> b n 1 1"
        )

    @g_timer.avg_time("rollout_storage.compute_returns", level=1)
    def compute_returns(self, next_value, use_gae, gamma, tau, clip_rewards: int = 0.0):
        rewards = self.buffers["rewards"]
        if clip_rewards > 0:
            rewards = torch.clamp(rewards, -clip_rewards, clip_rewards)

        if use_gae:
            assert isinstance(self.buffers["value_preds"], torch.Tensor)
            self.buffers["value_preds"][self.current_rollout_step_idx] = next_value
            gae = 0.0
            for step in reversed(range(self.current_rollout_step_idx)):
                delta = (
                    rewards[step]
                    + gamma
                    * self.buffers["value_preds"][step + 1]
                    * self.buffers["masks"][step + 1]
                    - self.buffers["value_preds"][step]
                )
                gae = delta + gamma * tau * gae * self.buffers["masks"][step + 1]
                self.buffers["returns"][step] = (  # type: ignore
                    gae + self.buffers["value_preds"][step]  # type: ignore
                )

        else:
            self.buffers["returns"][self.current_rollout_step_idx] = next_value
            for step in reversed(range(self.current_rollout_step_idx)):
                self.buffers["returns"][step] = (
                    gamma
                    * self.buffers["returns"][step + 1]
                    * self.buffers["masks"][step + 1]
                    + self.buffers["rewards"][step]
                )

    @g_timer.avg_time("ts.data_generator")
    def data_generator(
        self,
        advantages: Optional[torch.Tensor],
        num_mini_batch: int,
        norm_returns,
        norm_values,
    ) -> Iterator[DictTree]:
        assert isinstance(self.buffers["returns"], torch.Tensor)
        num_environments = self.buffers["returns"].size(1)
        assert num_environments >= num_mini_batch
        if num_environments % num_mini_batch != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_mini_batch
                )
            )

        yield from flatten_trajs(
            advantages,
            self.buffers,
            self._context_len,
            self.current_rollout_step_idx,
            self.hidden_window,
            self.before_start_att_masks,
            inds=torch.arange(num_environments),
            batch_size=num_environments // num_mini_batch,
            max_num_batches=num_mini_batch,
        )


def flatten_trajs(
    advantages,
    buffers,
    context_len: int,
    current_rollout_step_idx,
    hidden_window,
    before_start_att_masks,
    inds,
    batch_size: int,
    max_num_batches: int,
) -> TensorDict:
    max_data_window_size = current_rollout_step_idx
    device = advantages.device
    curr_slice = (
        slice(0, max_data_window_size),
        inds,
    )

    batch = buffers[curr_slice]
    batch["advantages"] = advantages[curr_slice]

    # Find where episode starts (when masks = False).
    dones = ~batch["masks"]
    seq_len = batch["masks"].shape[0]
    inserted_start = {}
    ep_starts = torch.nonzero(~batch["masks"])[:, :2]
    ep_starts_cpu = ep_starts.cpu().numpy().tolist()
    for batch_idx in range(len(inds)):
        if [0, batch_idx] not in ep_starts_cpu:
            inserted_start[(0, batch_idx)] = True
            ep_starts_cpu.insert(0, [0, batch_idx])
        current_index = 0
        while current_index < seq_len - context_len:
            ctx_dones_range = dones[
                current_index + 1 : current_index + context_len, batch_idx
            ]
            if torch.any(ctx_dones_range):
                first_start = torch.nonzero(ctx_dones_range)[0].min()
                first_start = int(first_start.item())
                current_index += first_start + 1
                if [current_index, batch_idx] not in ep_starts_cpu:
                    ep_starts_cpu.append([current_index, batch_idx])
            else:
                current_index += context_len
                if [current_index, batch_idx] not in ep_starts_cpu:
                    ep_starts_cpu.append([current_index, batch_idx])

    ep_starts_cpu = np.array(ep_starts_cpu)
    # Track the next start episode.
    batch_to_next_ep_starts = {}
    for batch_idx in range(len(inds)):
        batch_ep_starts = ep_starts_cpu[ep_starts_cpu[:, 1] == batch_idx][:, 0]
        batch_ep_starts = np.sort(batch_ep_starts)

        batch_to_next_ep_starts[batch_idx] = {}
        for i in range(len(batch_ep_starts) - 1):
            batch_to_next_ep_starts[batch_idx][batch_ep_starts[i]] = batch_ep_starts[
                i + 1
            ]

    chunked_ep_start_indices = torch.randperm(len(ep_starts_cpu)).numpy().tolist()
    num_valid_before_steps = before_start_att_masks.sum(-1).cpu().tolist()

    for batch_index in range(max_num_batches):
        eps = []
        fetch_before_counts = []
        while len(eps) < batch_size:
            if not chunked_ep_start_indices:
                chunked_ep_start_indices = (
                    torch.randperm(len(ep_starts_cpu)).numpy().tolist()
                )
            step_idx, batch_idx = ep_starts_cpu[chunked_ep_start_indices[0]]
            chunked_ep_start_indices = chunked_ep_start_indices[1:]

            next_ep_starts = batch_to_next_ep_starts[batch_idx]
            step_idx_end = next_ep_starts.get(step_idx, max_data_window_size)

            add_batch = batch.map(lambda x: x[step_idx:step_idx_end, batch_idx])

            did_insert_start = inserted_start.get((step_idx, batch_idx), False)
            if did_insert_start:
                fetch_before_counts.append(
                    max(num_valid_before_steps[batch_idx] - 1, 0)
                )
            else:
                fetch_before_counts.append(0)

            add_batch["observations"][ATT_MASK_K] = torch.ones(
                # The att mask needs to have same window size as the data batch.
                (step_idx_end - step_idx, 1),
                dtype=bool,
                device=device,
            )
            add_batch["observations"][START_HIDDEN_WINDOW_K] = hidden_window[
                batch_idx, : (context_len - 1)
            ]
            add_batch["observations"][START_ATT_MASK_K] = before_start_att_masks[
                batch_idx, :-1
            ].unsqueeze(-1)

            eps.append(add_batch)

        assert (
            len(eps) > 0
        ), "Collected no episodes from rollout, ensure episode horizon is shorter than rollout length."
        ret_batch = transpose_stack_pad_dicts(eps)

        ret_batch["observations"][FETCH_BEFORE_COUNTS_K] = torch.tensor(
            fetch_before_counts, device=device
        )

        yield ret_batch


def check_extracted_batch(add_batch, batch_idx):
    """
    Checks a batch is good. Note that the entire batch should be valid as we
    are going to attend over the entire sequence. Only the later stack pad
    operation will result in False attention masks.
    """

    debug_info = add_batch["observations"]["debug_info"]
    batch_step_info = debug_info[:, 0]
    batch_ep_info = debug_info[:, 1]
    # Each step should be increasing by 1.
    if not ((batch_step_info[:-1] + 1) == batch_step_info[1:]).all():
        raise ValueError(f"Episode steps not incrementing {batch_step_info}")

    # Steps should belong to the same episode.
    if not (batch_ep_info[0] == batch_ep_info).all():
        raise ValueError(f"Episode IDs not consistent")
