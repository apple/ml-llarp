#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import contextlib
import os
import os.path as osp
import random
import time
from collections import defaultdict, deque
from typing import Any, Optional

import cv2
import gym.spaces as spaces
import hydra
import numpy as np
import torch
import wandb
from habitat import logger
from habitat.config import read_write
from habitat.utils import profiling_wrapper
from habitat.utils.visualizations.utils import (observations_to_image,
                                                overlay_frame)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch, apply_obs_transforms_obs_space,
    get_active_obs_transforms)
from habitat_baselines.common.tensorboard_utils import get_writer
from habitat_baselines.rl.ddppo.ddp_utils import (EXIT, SAVE_STATE,
                                                  get_distrib_size,
                                                  is_slurm_batch_job,
                                                  load_resume_state,
                                                  rank0_only, requeue_job,
                                                  save_resume_state)
from habitat_baselines.rl.ddppo.policy import PointNavResNetNet
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer, get_device
from habitat_baselines.utils.common import (batch_obs, generate_video,
                                            inference_mode,
                                            is_continuous_action_space)
from habitat_baselines.utils.info_dict import (NON_SCALAR_METRICS,
                                               extract_scalars_from_info,
                                               extract_scalars_from_infos)
from habitat_baselines.utils.timing import g_timer
from omegaconf import OmegaConf
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from llarp.trainer.custom_ddp import init_distrib_slurm
from llarp.trainer.env_utils import GOAL_K, SUCCESS_K


def extract_scalars_from_infos_safe(
    infos: list[dict[str, Any]],
    ignore_keys: Optional[set[str]],
    n_envs: int,
    device,
) -> tuple[dict[str, Tensor], Tensor]:
    results = defaultdict(
        lambda: torch.zeros((n_envs, 1), device=device, dtype=torch.float)
    )
    is_valid = defaultdict(lambda: torch.zeros((n_envs, 1), device=device, dtype=bool))

    for i in range(len(infos)):
        for k, v in extract_scalars_from_info(infos[i], ignore_keys).items():
            results[k][i, 0] = v
            is_valid[k][i] = True

    return results, is_valid


@baseline_registry.register_trainer(name="il_ppo")
class ILPPOTrainer(PPOTrainer):
    def _should_save_resume_state(self) -> bool:
        return SAVE_STATE.is_set() or (
            (
                not self.config.habitat_baselines.rl.preemption.save_state_batch_only
                or is_slurm_batch_job()
            )
            and (
                (
                    int(self.num_updates_done + 1)
                    % self.config.habitat_baselines.rl.preemption.save_resume_state_interval
                )
                == 0
            )
        )

    def should_checkpoint(self) -> bool:
        needs_checkpoint = False
        if self.config.habitat_baselines.num_checkpoints != -1:
            checkpoint_every = 1 / self.config.habitat_baselines.num_checkpoints
            if self._last_checkpoint_percent + checkpoint_every <= self.percent_done():
                needs_checkpoint = True
                self._last_checkpoint_percent = self.percent_done()
        else:
            needs_checkpoint = (
                self.num_updates_done
                % self.config.habitat_baselines.checkpoint_interval
            ) == 0

        return needs_checkpoint

    def _coalesce_post_step(self, losses, count_steps_delta: int):
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack([self.running_episode_stats[k] for k in stats_ordering], 0)

        #######################################################################
        # ADDED
        # Sum over the environment dimension. This doesn't change the outcome
        # since we later sum over counts and metric values to compute averages.
        stats = stats.sum(1, keepdims=True)
        #######################################################################

        stats = self._all_reduce(stats)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        if self._is_distributed:
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            stats /= torch.distributed.get_world_size()

            losses = {k: stats[i].item() for i, k in enumerate(loss_name_ordering)}

        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses

    def train(self) -> None:
        resume_state = load_resume_state(self.config)
        self._init_train(resume_state)

        # Setting these to -1 will checkpoint/render on the 1st step. Setting
        # to 0.0 will skip the 1st step.
        self._last_checkpoint_percent = 0.0
        self._last_render_percent = 0.0

        count_checkpoints = 0
        prev_time = 0
        self._num_render = 0
        self._render_idx = 0
        self._frames = defaultdict(list)
        self._prev_dones = None

        if self._is_distributed:
            torch.distributed.barrier()

        resume_run_id = None
        if resume_state is not None:
            self._agent.load_state_dict(resume_state)

            requeue_stats = resume_state["requeue_stats"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats["_last_checkpoint_percent"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(requeue_stats["window_episode_stats"])
            resume_run_id = requeue_stats.get("run_id", None)

        if (
            hasattr(self.envs, "action_decode_tree")
            and self.envs.action_decode_tree is not None
        ):
            self._agent.actor_critic._policy_core.set_action_decode_tree(
                self.envs.action_decode_tree
            )

        with (
            get_writer(
                self.config,
                resume_run_id=resume_run_id,
                flush_secs=self.flush_secs,
                purge_step=int(self.num_steps_done),
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                self._agent.pre_rollout()

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                        run_id=writer.get_run_id(),
                    )

                    save_resume_state(
                        dict(
                            **self._agent.get_resume_state(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    self.envs.close()

                    requeue_job()

                    return

                self._agent.eval()
                count_steps_delta = 0

                if self._ppo_cfg.num_steps != 0:
                    count_steps_delta = self._rollout(count_steps_delta)
                else:
                    count_steps_delta += 1

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                losses = self._update_agent()

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    losses,
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                if self.config.habitat_baselines.num_render_batches != -1:
                    render_every = 1 / self.config.habitat_baselines.num_render_batches
                    if self._last_render_percent + render_every < self.percent_done():
                        self._last_render_percent = self.percent_done()
                        self._num_render = self.config.habitat_baselines.num_render
                        self._render_idx += 1

                # Debug code to render every step.
                # self._num_render = self.config.habitat_baselines.num_render
                # self._render_idx += 1

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

            self.envs.close()

    def _flush_frames(self, is_done, full_info, env_i):
        """
        Write the frames to wandb.
        """

        if not is_done:
            return

        if self.config.habitat_baselines.writer_type == "wb" and rank0_only():
            if SUCCESS_K in full_info:
                frames = [
                    _add_text_at_bottom(frame, str(full_info[SUCCESS_K]))
                    for frame in self._frames[env_i]
                ]
            else:
                frames = self._frames[env_i]
            # Upload the generated video to wandb.
            frames = np.stack(frames, axis=0)
            frames = np.transpose(frames, (0, 3, 1, 2))

            wandb.log(
                {
                    f"video_{self._num_render}": wandb.Video(
                        frames, fps=self.config.habitat_baselines.video_fps
                    )
                },
                step=int(self.num_steps_done),
            )

        self._num_render -= 1
        self._frames[env_i] = []
        if self._num_render == 0:
            self._frames = defaultdict(list)

    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        with g_timer.avg_time("trainer.step_env"):
            outputs = [
                self.envs.wait_step_at(index_env)
                for index_env in range(env_slice.start, env_slice.stop)
            ]

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]

        if self._num_render > 0 and rank0_only():
            for env_i in range(len(rewards_l)):
                if (
                    self._prev_dones is not None
                    and len(self._frames[env_i]) == 0
                    and not self._prev_dones[env_i]
                ):
                    # We can't start rendering in the middle of a trajectory.
                    # If the last step is `done`, then we are at the start of a
                    # new trajectory.
                    continue

                full_info = {
                    "reward": rewards_l[env_i],
                    # Don't log because there will be too many sub-task info elements.
                    # **infos[env_i],
                }

                # Only select some keys to show.
                if SUCCESS_K in infos[env_i]:
                    full_info[SUCCESS_K] = infos[env_i][SUCCESS_K]
                if GOAL_K in infos[env_i]:
                    full_info[GOAL_K] = infos[env_i][GOAL_K]
                if "num_steps" in infos[env_i]:
                    full_info["num_steps"] = infos[env_i]["num_steps"]
                frame = observations_to_image(
                    {k: v for k, v in observations[env_i].items()},
                    full_info,
                )

                frame = overlay_frame(frame, full_info)
                self._frames[env_i].append(frame)
                self._flush_frames(dones[env_i], full_info, env_i)
                if self._num_render == 0:
                    break
        self._prev_dones = dones

        with g_timer.avg_time("trainer.update_stats"):
            observations = self.envs.post_step(observations)
            batch = batch_obs(observations, device=self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            rewards = torch.tensor(
                rewards_l,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            )
            rewards = rewards.unsqueeze(1)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device=self.current_episode_reward.device,
            )
            done_masks = torch.logical_not(not_done_masks)

            self.current_episode_reward[env_slice] += rewards
            current_ep_reward = self.current_episode_reward[env_slice]
            self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
            self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore

            self._single_proc_infos = extract_scalars_from_infos(
                infos,
                ignore_keys=set(
                    k for k in infos[0].keys() if k not in self._rank0_keys
                ),
            )
            extracted_infos, is_valids = extract_scalars_from_infos_safe(
                infos,
                self._rank0_keys,
                done_masks.shape[0],
                self.current_episode_reward.device,
            )

            for k, v in extracted_infos.items():
                is_valid = is_valids[k]
                should_write = torch.logical_and(done_masks, is_valid)

                if k not in self.running_episode_stats:
                    self.running_episode_stats[k] = torch.zeros_like(
                        self.running_episode_stats["reward"]
                    )
                count_k = f"{k}_count"
                if count_k not in self.running_episode_stats:
                    self.running_episode_stats[count_k] = torch.zeros_like(
                        self.running_episode_stats["reward"]
                    )

                self.running_episode_stats[count_k][env_slice] += should_write.float()  # type: ignore
                self.running_episode_stats[k][env_slice] += v.where(should_write, v.new_zeros(()))  # type: ignore

            self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)

        if self._is_static_encoder:
            with inference_mode(), g_timer.avg_time("trainer.visual_features"):
                batch[PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY] = self._encoder(
                    batch
                )

        self._agent.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self._agent.rollouts.advance_rollout(buffer_index)

        return env_slice.stop - env_slice.start

    @rank0_only
    def _training_log(self, writer, losses: dict[str, float], prev_time: int = 0):
        deltas = {
            k: ((v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item())
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas[f"{k}_count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
            and not k.endswith("_count")
            and deltas[f"{k}_count"] > 0
        }
        # Only log metrics we are actually collecting data for.
        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)

        # Difference from default trainer: This will only log the wandb stats every so often. But we want to log on the first step so the plots start at 0.
        if (
            self.num_updates_done - 1
        ) % self.config.habitat_baselines.log_interval == 0:
            for k, v in metrics.items():
                writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
            for k, v in losses.items():
                writer.add_scalar(f"learner/{k}", v, self.num_steps_done)

            for k, v in self._single_proc_infos.items():
                writer.add_scalar(k, np.mean(v), self.num_steps_done)

            writer.add_scalar(
                "reward",
                deltas["reward"] / deltas["count"],
                self.num_steps_done,
            )
            # Log perf metrics.
            writer.add_scalar("perf/fps", fps, self.num_steps_done)

            for timer_name, timer_val in g_timer.items():
                writer.add_scalar(
                    f"perf/{timer_name}",
                    timer_val.mean,
                    self.num_steps_done,
                )

        # log stats
        logger.info(
            "update: {}\tfps: {:.3f}\t".format(
                self.num_updates_done,
                fps,
            )
        )

        logger.info(
            f"Num updates: {self.num_updates_done}\tNum frames {self.num_steps_done}"
        )

        logger.info(
            "Average window size: {}  {}".format(
                len(self.window_episode_stats["count"]),
                "  ".join("{}: {:.3f}".format(k, v) for k, v in metrics.items()),
            )
        )

        # perf_stats_str = " ".join([f"{k}: {v.mean:.3f}" for k, v in g_timer.items()])
        # logger.info(f"\tPerf Stats: {perf_stats_str}")
        # if self.config.habitat_baselines.should_log_single_proc_infos:
        #     for k, v in self._single_proc_infos.items():
        #         logger.info(f" - {k}: {np.mean(v):.3f}")

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        """
        Modified over the base implementation to support the multi-discrete action space.
        """

        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._agent.nbuffers),
            int((buffer_index + 1) * num_envs / self._agent.nbuffers),
        )

        with g_timer.avg_time("trainer.sample_action"), inference_mode():
            # Sample actions
            step_batch = self._agent.rollouts.get_current_step(env_slice, buffer_index)

            # Obtain lenghts
            step_batch_lens = {
                k: v for k, v in step_batch.items() if k.startswith("index_len")
            }
            action_data = self._agent.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
                **step_batch_lens,
            )

        with g_timer.avg_time("trainer.obs_insert"):
            for index_env, act in zip(
                range(env_slice.start, env_slice.stop),
                action_data.env_actions.cpu().unbind(0),
            ):
                if is_continuous_action_space(self._env_spec.action_space):
                    if self.config.habitat_baselines.should_clip_cont_action:
                        # Clipping actions to the specified limits
                        act = np.clip(
                            act.numpy(),
                            self._env_spec.action_space.low,
                            self._env_spec.action_space.high,
                        )
                    else:
                        act = act.numpy()
                elif isinstance(self._env_spec.action_space, spaces.MultiDiscrete):
                    act = act.numpy()
                else:
                    act = act.item()
                self.envs.async_step_at(index_env, act)

        with g_timer.avg_time("trainer.obs_insert"):
            self._agent.rollouts.insert(
                next_recurrent_hidden_states=action_data.rnn_hidden_states,
                actions=action_data.actions,
                action_log_probs=action_data.action_log_probs,
                value_preds=action_data.values,
                buffer_index=buffer_index,
                should_inserts=action_data.should_inserts,
                action_data=action_data,
            )

    @g_timer.avg_time("trainer.update_agent")
    def _update_agent(self):
        if self._ppo_cfg.num_steps != 0:
            with inference_mode():
                step_batch = self._agent.rollouts.get_last_step()
                next_value = self._agent.actor_critic.get_value(
                    step_batch["observations"],
                    step_batch.get("recurrent_hidden_states", None),
                    step_batch["prev_actions"],
                    step_batch["masks"],
                )

            self._agent.rollouts.compute_returns(
                next_value,
                self._ppo_cfg.use_gae,
                self._ppo_cfg.gamma,
                self._ppo_cfg.tau,
                self._ppo_cfg.clip_rewards,
            )

        self._agent.train()

        losses = self._agent.updater.update(self._agent.rollouts)

        self._agent.rollouts.after_update()
        self._agent.after_update()

        return losses

    @g_timer.avg_time("trainer.rollout_collect")
    def _rollout(self, count_steps_delta):
        for buffer_index in range(self._agent.nbuffers):
            self._compute_actions_and_step_envs(buffer_index)

        for step in range(self._ppo_cfg.num_steps):
            is_last_step = (
                self.should_end_early(step + 1) or (step + 1) == self._ppo_cfg.num_steps
            )

            for buffer_index in range(self._agent.nbuffers):
                count_steps_delta += self._collect_environment_result(buffer_index)

                if not is_last_step:
                    self._compute_actions_and_step_envs(buffer_index)

            if is_last_step:
                break
        return count_steps_delta

    def _init_train(self, resume_state=None):
        if resume_state is None:
            resume_state = load_resume_state(self.config)

        if resume_state is not None:
            if not self.config.habitat_baselines.load_resume_state_config:
                raise FileExistsError(
                    f"The configuration provided has habitat_baselines.load_resume_state_config=False but a previous training run exists. You can either delete the checkpoint folder {self.config.habitat_baselines.checkpoint_folder}, or change the configuration key habitat_baselines.checkpoint_folder in your new run."
                )

            self.config = self._get_resume_state_config_or_new_config(
                resume_state["config"]
            )

        if self.config.habitat_baselines.rl.ddppo.force_distributed:
            self._is_distributed = True

        self._add_preemption_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.habitat_baselines.rl.ddppo.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            with read_write(self.config):
                self.config.habitat_baselines.torch_gpu_id = local_rank
                self.config.habitat.simulator.habitat_sim_v0.gpu_device_id = local_rank
                # Multiply by the number of simulators to make sure they also get unique seeds
                self.config.habitat.seed += (
                    torch.distributed.get_rank()
                    * self.config.habitat_baselines.num_environments
                )

            random.seed(self.config.habitat.seed)
            np.random.seed(self.config.habitat.seed)
            torch.manual_seed(self.config.habitat.seed)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.habitat_baselines.verbose:
            logger.info(f"config: {OmegaConf.to_yaml(self.config)}")

        profiling_wrapper.configure(
            capture_start_step=self.config.habitat_baselines.profiling.capture_start_step,
            num_steps_to_capture=self.config.habitat_baselines.profiling.num_steps_to_capture,
        )

        # remove the non scalar measures from the measures since they can only be used in
        # evaluation
        for non_scalar_metric in NON_SCALAR_METRICS:
            non_scalar_metric_root = non_scalar_metric.split(".")[0]
            if non_scalar_metric_root in self.config.habitat.task.measurements:
                with read_write(self.config):
                    OmegaConf.set_struct(self.config, False)
                    self.config.habitat.task.measurements.pop(non_scalar_metric_root)
                    OmegaConf.set_struct(self.config, True)
                if self.config.habitat_baselines.verbose:
                    logger.info(
                        f"Removed metric {non_scalar_metric_root} from metrics since it cannot be used during training."
                    )

        self._init_envs()

        self.device = get_device(self.config)

        if rank0_only() and not os.path.isdir(
            self.config.habitat_baselines.checkpoint_folder
        ):
            os.makedirs(self.config.habitat_baselines.checkpoint_folder)

        log_dir = osp.dirname(self.config.habitat_baselines.log_file)
        if len(log_dir) > 0:
            os.makedirs(log_dir, exist_ok=True)
        logger.add_filehandler(self.config.habitat_baselines.log_file)

        self._agent = self._create_agent(resume_state)
        if self._is_distributed:
            self._agent.updater.init_distributed(find_unused_params=False)  # type: ignore
        self._agent.post_init()

        self._is_static_encoder = (
            not self.config.habitat_baselines.rl.ddppo.train_encoder
        )
        self._ppo_cfg = self.config.habitat_baselines.rl.ppo

        observations = self.envs.reset()
        observations = self.envs.post_step(observations)
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._is_static_encoder:
            self._encoder = self._agent.actor_critic.visual_encoder
            assert (
                self._encoder is not None
            ), "Visual encoder is not specified for this actor"
            with inference_mode():
                batch[PointNavResNetNet.PRETRAINED_VISUAL_FEATURES_KEY] = self._encoder(
                    batch
                )

        self._agent.rollouts.insert_first_observations(batch)

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=self._ppo_cfg.reward_window_size)
        )

        self.t_start = time.time()


def _add_text_at_bottom(frame, txt):
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    textsize = cv2.getTextSize(txt, font, font_size, font_thickness)[0]

    x = 10
    y = frame.shape[0] - textsize[1]

    cv2.putText(
        frame,
        txt,
        (x, y),
        font,
        font_size,
        (0, 0, 0),
        font_thickness * 2,
        lineType=cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        txt,
        (x, y),
        font,
        font_size,
        (255, 255, 255, 255),
        font_thickness,
        lineType=cv2.LINE_AA,
    )
    return np.clip(frame, 0, 255)
