#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import abc
import os
import os.path as osp
import pickle
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

import gym.spaces as spaces
import numpy as np
import torch
import tqdm
from habitat import VectorEnv, logger
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils.visualizations.utils import (observations_to_image,
                                                overlay_frame)
from habitat_baselines.common.env_spec import EnvironmentSpec
from habitat_baselines.common.obs_transformers import (
    ObservationTransformer, apply_obs_transforms_batch,
    apply_obs_transforms_obs_space, get_active_obs_transforms)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ppo.evaluator import Evaluator, pause_envs
from habitat_baselines.rl.ppo.policy import Policy
from habitat_baselines.utils.common import (batch_obs, generate_video,
                                            get_action_space_info,
                                            inference_mode,
                                            is_continuous_action_space)
from habitat_baselines.utils.info_dict import extract_scalars_from_info
from numpy import ndarray
from torch import Tensor

from llarp.dataset.demo_dataset import DatasetCollector, DemoDataset, TrajInfo
from llarp.policies import LlmPolicy
from llarp.task.measures import LangGoalMeasure
from llarp.task.sensors import StepCountSensor


class CustomHabitatEvaluator(Evaluator):
    def evaluate_agent(
        self,
        agent,
        envs,
        config,
        checkpoint_index,
        step_id,
        writer,
        device,
        obs_transforms,
        env_spec,
        rank0_keys,
    ):
        if isinstance(agent.actor_critic, LlmPolicy) and hasattr(
            envs, "action_decode_tree"
        ):
            agent.actor_critic._policy_core.set_action_decode_tree(
                envs.action_decode_tree
            )

        os.makedirs(config.habitat_baselines.checkpoint_folder, exist_ok=True)
        observations = envs.reset()
        observations = envs.post_step(observations)
        batch = batch_obs(observations, device=device)
        batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

        action_shape, discrete_actions = get_action_space_info(
            agent.actor_critic.policy_action_space
        )

        current_episode_reward = torch.zeros(envs.num_envs, 1, device="cpu")
        rnn_hxs_dim = agent.actor_critic.recurrent_hidden_size
        if not isinstance(rnn_hxs_dim, tuple):
            rnn_hxs_dim = (rnn_hxs_dim,)

        test_recurrent_hidden_states = torch.zeros(
            (
                config.habitat_baselines.num_environments,
                agent.actor_critic.num_recurrent_layers,
                *rnn_hxs_dim,
            ),
            device=device,
        )
        should_update_recurrent_hidden_states = (
            np.prod(test_recurrent_hidden_states.shape) != 0
        )
        prev_actions = torch.zeros(
            config.habitat_baselines.num_environments,
            *action_shape,
            device=device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            config.habitat_baselines.num_environments,
            1,
            device=device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        rgb_frames: List[List[np.ndarray]] = [
            [observations_to_image({k: v[env_idx] for k, v in batch.items()}, {})]
            for env_idx in range(config.habitat_baselines.num_environments)
        ]

        if len(config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(config.habitat_baselines.video_dir, exist_ok=True)

        evals_per_ep = config.habitat_baselines.eval.evals_per_ep
        number_of_eval_episodes = config.habitat_baselines.test_episode_count
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(envs.number_of_episodes)
        else:
            total_num_eps = sum(envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    f", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        # Track demo data.
        n_envs = config.habitat_baselines.num_environments

        if config.habitat_baselines.eval_demo_save_name is not None:
            demo_collector = DatasetCollector(
                config.habitat_baselines.num_environments,
                config.habitat_baselines.eval_save_demos_flush_interval,
                config.habitat_baselines.eval_demo_save_name,
            )
        else:
            demo_collector = None

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        agent.eval()
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and envs.num_envs > 0
        ):
            current_episodes_info = envs.current_episodes()

            if demo_collector is not None:
                demo_collector.collect_obs(batch)

            with inference_mode():
                action_data = agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = action_data.rnn_hidden_states
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    for i, should_insert in enumerate(action_data.should_inserts):
                        if not should_insert.item():
                            continue
                        if should_update_recurrent_hidden_states:
                            test_recurrent_hidden_states[
                                i
                            ] = action_data.rnn_hidden_states[i]
                        prev_actions[i].copy_(action_data.actions[i])  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        env_spec.action_space.low,
                        env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            elif isinstance(env_spec.action_space, spaces.MultiDiscrete):
                step_data = action_data.env_actions.cpu().numpy()
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            outputs = envs.step(step_data)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            # Note that `policy_infos` represents the information about the
            # action BEFORE `observations` (the action used to transition to
            # `observations`).
            policy_infos = agent.actor_critic.get_extra(action_data, infos, dones)
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            observations = envs.post_step(observations)
            batch = batch_obs(  # type: ignore
                observations,
                device=device,
            )
            batch = apply_obs_transforms_batch(batch, obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = envs.current_episodes()
            envs_to_pause = []
            n_envs = envs.num_envs
            if demo_collector is not None:
                demo_collector.collect_action(action_data.actions, infos)
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)

                # Exclude the keys from `_rank0_keys` from displaying in the video
                disp_info = {
                    k: v
                    for k, v in infos[i].items()
                    if k not in rank0_keys and type(v) != list
                }
                frame_info = dict(disp_info)
                if config.habitat_baselines.video_mode:
                    rename_map = {
                        "lang_goal": "Task Goal",
                        "pddl_action": "Policy Action",
                    }
                    frame_info = {
                        rename_map.get(k, k): v for k, v in frame_info.items()
                    }

                if len(config.habitat_baselines.eval.video_option) > 0:
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, disp_info
                    )
                    if not not_done_masks[i].item():
                        final_frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()},
                            disp_info,
                        )
                        final_frame = overlay_frame(final_frame, frame_info)
                        rgb_frames[i].append(final_frame)
                        # The starting frame of the next episode will be the final element..
                        rgb_frames[i].append(frame)
                    else:
                        frame = overlay_frame(frame, frame_info)
                        rgb_frames[i].append(frame)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(
                        extract_scalars_from_info(
                            # We can't extract string types from the logged info.
                            {
                                k: v
                                for k, v in infos[i].items()
                                if not isinstance(v, list)
                            }
                        )
                    )

                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if demo_collector is not None:
                        demo_collector.on_ep_done(i, infos[i])

                    if len(config.habitat_baselines.eval.video_option) > 0:
                        generate_video(
                            video_option=config.habitat_baselines.eval.video_option,
                            video_dir=config.habitat_baselines.video_dir,
                            # Since the final frame is the start frame of the next episode.
                            images=rgb_frames[i][:-1],
                            episode_id=f"{current_episodes_info[i].episode_id}_{ep_eval_count[k]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(disp_info),
                            fps=config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        # Since the starting frame of the next episode is the final frame.
                        rgb_frames[i] = rgb_frames[i][-1:]

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            if test_recurrent_hidden_states is None:
                test_recurrent_hidden_states = torch.zeros(not_done_masks.shape)
            not_done_masks = not_done_masks.to(device=device)

            if config.habitat_baselines.eval_allow_env_pauses:
                (
                    envs,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    current_episode_reward,
                    prev_actions,
                    batch,
                    rgb_frames,
                ) = pause_envs(
                    envs_to_pause,
                    envs,
                    test_recurrent_hidden_states,
                    not_done_masks,
                    current_episode_reward,
                    prev_actions,
                    batch,
                    rgb_frames,
                )

        pbar.close()

        aggregated_stats = {}
        all_ks = set()
        for ep in stats_episodes.values():
            all_ks.update(ep.keys())

        log_fname = osp.join(
            "data/logs", config.habitat_baselines.wb.run_name + "_results.pickle"
        )
        with open(log_fname, "wb") as f:
            pickle.dump(stats_episodes, f)

        for stat_key in all_ks:
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values() if stat_key in v]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        if demo_collector is not None:
            demo_collector.flush()
