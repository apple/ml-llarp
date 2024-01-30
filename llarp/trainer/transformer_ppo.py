#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from habitat.core.logging import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.algo.ddppo import (_cpu_to_numpy,
                                                   _EvalActionsWrapper,
                                                   distributed_var_mean)
from habitat_baselines.rl.ppo.policy import NetPolicy
from habitat_baselines.rl.ppo.ppo import PPO
from habitat_baselines.rl.ver.ver_rollout_storage import VERRolloutStorage
from habitat_baselines.utils.common import (LagrangeInequalityCoefficient,
                                            inference_mode)
from habitat_baselines.utils.timing import g_timer
from torch import Tensor
from transformers import get_cosine_schedule_with_warmup

from llarp.policies.transformer_storage import (ATT_MASK_K,
                                                FETCH_BEFORE_COUNTS_K)

EPS_PPO = 1e-5


def record_min_mean_max(t: torch.Tensor, prefix: str, learner_metrics):
    for name, op in (
        ("min", torch.min),
        ("mean", torch.mean),
        ("max", torch.max),
    ):
        learner_metrics[f"{prefix}_{name}"].append(op(t))


@baseline_registry.register_updater
class TransformerPPO(PPO):
    """
    Custom PPO implementation to handle the transformer-based policy window
    contexts.
    """

    @classmethod
    def from_config(cls, actor_critic: NetPolicy, config):
        return cls(
            actor_critic=actor_critic,
            clip_param=config.clip_param,
            ppo_epoch=config.ppo_epoch,
            num_mini_batch=config.num_mini_batch,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            max_grad_norm=config.max_grad_norm,
            use_clipped_value_loss=config.use_clipped_value_loss,
            use_normalized_advantage=config.use_normalized_advantage,
            entropy_target_factor=config.entropy_target_factor,
            use_adaptive_entropy_pen=config.use_adaptive_entropy_pen,
            config=config,
        )

    def __init__(
        self,
        *args,
        use_adaptive_entropy_pen,
        lr,
        eps,
        entropy_target_factor,
        num_mini_batch,
        ppo_epoch,
        config,
        **kwargs,
    ):
        # We will manually set the adaptive entropy penalty.
        super().__init__(
            *args,
            use_adaptive_entropy_pen=False,
            lr=lr,
            eps=eps,
            entropy_target_factor=entropy_target_factor,
            num_mini_batch=num_mini_batch,
            ppo_epoch=ppo_epoch,
            **kwargs,
        )

        if use_adaptive_entropy_pen:
            self.entropy_coef = LagrangeInequalityCoefficient(
                -float(entropy_target_factor),
                init_alpha=self.entropy_coef,
                alpha_max=1.0,
                alpha_min=1e-4,
                greater_than=True,
            ).to(device=self.device)

        # Create here so the entropy coef parameter can be registered if we are
        # doing auto entropy coef.
        params = list(filter(lambda p: p[1].requires_grad, self.named_parameters()))
        print(
            f"Number of params to train: {sum(param.numel() for _, param in params):,}"
        )

        if config.llm_lr_scaling == 1.0:
            self.optimizer = optim.AdamW([p for _, p in params], lr=lr, eps=eps)
        else:
            llm_params = []
            non_llm_params = []
            for n, p in params:
                if "._llm" in n:
                    llm_params.append(p)
                else:
                    non_llm_params.append(p)

            self.optimizer = optim.AdamW(
                [
                    {"params": non_llm_params},
                    {"params": llm_params, "lr": lr * config.llm_lr_scaling},
                ],
                lr=lr,
                eps=eps,
            )

        if config.use_cosine_lr_decay:
            if config.num_steps == 0:
                # Only IL.
                n_updates = int(config.total_num_steps)
            else:
                n_updates = int(config.total_num_steps) // (
                    config.num_steps * config.num_envs
                )
            self._lr_scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_training_steps=n_updates,
                num_warmup_steps=config.n_warmup_steps,
            )
        else:
            self._lr_scheduler = None

        self.use_huber_loss = config.use_huber_loss

    def after_update(self) -> None:
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

    def _create_optimizer(self, lr, eps):
        # Hack so we can create the optimizer outside of the base PPO init.
        return None

    def _get_rl_loss(self, batch, learner_metrics, epoch, rollouts):
        (
            values,
            action_log_probs,
            dist_entropy,
            aux_loss_res,
        ) = self._evaluate_actions(
            batch["observations"],
            batch["actions"],
        )

        ratio = torch.exp(action_log_probs - batch["action_log_probs"])

        surr1 = batch["advantages"] * ratio
        surr2 = batch["advantages"] * (
            torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
        )
        action_loss = -torch.min(surr1, surr2)
        device = action_loss.device
        batch_size, context_len, _ = action_loss.shape

        # Apply VF normalization to this data batch.
        critic = self.actor_critic._critic

        # Returns and values are in the same normalized space.
        values = values.float()
        orig_values = values
        returns = batch["returns"]
        batch_value_preds = batch["value_preds"]

        if self.use_huber_loss:
            value_loss = F.huber_loss(
                values,
                returns,
                reduction="none",
                delta=1,
            )
        elif self.use_clipped_value_loss:
            delta = values.detach() - batch_value_preds
            value_pred_clipped = batch_value_preds + delta.clamp(
                -self.clip_param, self.clip_param
            )
            values = torch.where(
                delta.abs() < self.clip_param,
                values,
                value_pred_clipped,
            )
            value_loss = 0.5 * F.mse_loss(values, returns, reduction="none")
        else:
            value_loss = 0.5 * F.mse_loss(values, returns, reduction="none")

        loss_mask = batch["observations"][ATT_MASK_K]
        n_samples = loss_mask.sum()

        def att_reduce_mean(loss):
            return (loss * loss_mask).sum() / n_samples

        action_loss, value_loss, dist_entropy = map(
            att_reduce_mean,
            (action_loss, value_loss, dist_entropy),
        )
        all_losses = [
            self.value_loss_coef * value_loss,
            action_loss,
        ]

        if isinstance(self.entropy_coef, float):
            all_losses.append(-self.entropy_coef * dist_entropy)
        else:
            all_losses.append(self.entropy_coef.lagrangian_loss(dist_entropy))

        all_losses.extend(v["loss"] for v in aux_loss_res.values())

        with inference_mode():
            learner_metrics["value_pred_mean"].append(att_reduce_mean(orig_values))
            learner_metrics["prob_ratio_mean"].append(att_reduce_mean(ratio))
            learner_metrics["returns_mean"].append(att_reduce_mean(returns))

            learner_metrics["value_loss"].append(value_loss)
            learner_metrics["action_loss"].append(action_loss)
            learner_metrics["dist_entropy"].append(dist_entropy)
            if epoch == (self.ppo_epoch - 1):
                above_clipped = att_reduce_mean(
                    (ratio > (1.0 + self.clip_param)).float()
                )
                below_clipped = att_reduce_mean(
                    (ratio < (1.0 - self.clip_param)).float()
                )
                learner_metrics["ppo_fraction_clipped"].append(
                    above_clipped + below_clipped
                )
            for name, res in aux_loss_res.items():
                for k, v in res.items():
                    learner_metrics[f"aux_{name}_{k}"].append(v.detach())
            for name, val in self.actor_critic.policy_core_log.items():
                learner_metrics[f"policy.{name}"].append(val)

            total_n_this_batch = np.prod(batch["masks"].shape)

            max_n_per_batch = rollouts.num_steps * (
                rollouts._num_envs // self.num_mini_batch
            )
            n_this_batch = loss_mask.sum().item()

            learner_metrics["batch_util_ratio"].append(n_this_batch / max_n_per_batch)
            learner_metrics["eff_batch_size"].append(n_this_batch)
            learner_metrics["batch_nopad_ratio"].append(
                n_this_batch / total_n_this_batch
            )

            if self._lr_scheduler is not None:
                # There are multiple param groups. The 1st one is the higher LR
                # group, only report that.
                learner_metrics["lr"].append(self._lr_scheduler.get_last_lr()[0])

        return all_losses

    def get_advantages(self, rollouts) -> Tensor:
        advantages = rollouts.buffers["returns"] - rollouts.buffers["value_preds"]
        if not self.use_normalized_advantage:
            return advantages

        # Adjusted for multi-GPU training.
        var, mean = self._compute_var_mean(advantages[torch.isfinite(advantages)])

        advantages -= mean

        return advantages.mul_(torch.rsqrt(var + EPS_PPO))

    def update(
        self,
        rollouts,
    ):
        critic = self.actor_critic._critic
        all_obs = rollouts.buffers["observations"]

        # Convert the value predictions back to normalized scores. Convert
        # under the CURRENT normalization statistics to undue the normalization
        # effect. If we did this aafter the update, then it would be
        # denormalized with different normalization statistics.
        norm_value_preds = critic.post_process_returns(
            rollouts.buffers["value_preds"], all_obs
        )

        # Update the normalization statistics (this will also update the linear projection of the network.
        with torch.no_grad():
            critic.update_stats(rollouts.buffers["returns"], all_obs)

        # Compute advantages with unnormalized returns since the value preds
        # are also unnormalized and we can separately normalize the difference.
        advantages = self.get_advantages(rollouts)

        # Normalize the returns under the updated statistics.
        norm_returns = critic.post_process_returns(rollouts.buffers["returns"], all_obs)

        learner_metrics = collections.defaultdict(list)

        if rollouts.num_steps == 0:
            # Only IL
            self._update_from_batch(None, 0, rollouts, learner_metrics)
            logger.info(f"IL Loss: {learner_metrics['il_loss']}")
        else:
            for epoch in range(self.ppo_epoch):
                data_generator = rollouts.data_generator(
                    advantages, self.num_mini_batch, norm_returns, norm_value_preds
                )

                for _bid, batch in enumerate(data_generator):
                    self._update_from_batch(batch, epoch, rollouts, learner_metrics)

        self._set_grads_to_none()

        with inference_mode():
            return {
                k: float(
                    torch.stack(
                        [torch.as_tensor(v, dtype=torch.float32) for v in vs]
                    ).mean()
                )
                for k, vs in learner_metrics.items()
            }

    @g_timer.avg_time("ppo.update_from_batch", level=1)
    def _update_from_batch(self, batch, epoch, rollouts, learner_metrics):
        """
        Performs a gradient update from the minibatch.
        """
        self._set_grads_to_none()
        self.optimizer.zero_grad()

        all_losses = []
        if rollouts.num_steps != 0:
            all_losses.extend(
                self._get_rl_loss(batch, learner_metrics, epoch, rollouts)
            )

        total_loss = torch.stack(all_losses).sum()

        total_loss = self.before_backward(total_loss)
        total_loss.backward()
        self.after_backward(total_loss)

        grad_norm = self.before_step()
        self.optimizer.step()
        self.after_step()

        with inference_mode():
            learner_metrics["grad_norm"].append(grad_norm)
            if isinstance(self.entropy_coef, LagrangeInequalityCoefficient):
                learner_metrics["entropy_coef"].append(self.entropy_coef().detach())


class ModelParallelDecentralizedDistributedMixin:
    @staticmethod
    def _compute_var_mean(x):
        return distributed_var_mean(x)

    def init_distributed(self, find_unused_params: bool = True) -> None:
        class Guard:  # noqa: SIM119
            def __init__(self, model, device):
                if device.type == "cuda":
                    torch.cuda.set_device(device)
                    self.ddp = torch.nn.parallel.DistributedDataParallel(  # type: ignore
                        model,
                        # MUST NOT SPECIFY THESE FOR MODEL PARALLEL.
                        # device_ids=[device],
                        # output_device=device,
                        find_unused_parameters=find_unused_params,
                        static_graph=True,
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(  # type: ignore
                        model,
                        find_unused_parameters=find_unused_params,
                    )

        self._evaluate_actions_wrapper = Guard(_EvalActionsWrapper(self.actor_critic), self.device)  # type: ignore

    def _evaluate_actions(self, *args, **kwargs):
        return self._evaluate_actions_wrapper.ddp(
            *_cpu_to_numpy(args), **_cpu_to_numpy(kwargs)
        )


@baseline_registry.register_updater
class DistributedTransformerPPO(
    ModelParallelDecentralizedDistributedMixin, TransformerPPO
):
    pass
