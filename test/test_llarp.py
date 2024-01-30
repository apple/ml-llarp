#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""
Tests to ensure that LLaRP is training correctly.
"""

import glob
import os
import random

import hydra
import numpy as np
import pytest
import torch
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat_baselines.common.baseline_registry import baseline_registry

import llarp.dataset
import llarp.policies
import llarp.task
import llarp.trainer
from llarp.task.predicate_task import RearrangePredicateTask
from llarp.trainer.test_env_factory import TestBatchedEnv


def pre_config_setup():
    os.environ["HABITAT_ENV_DEBUG"] = "1"

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path="../llarp/config/", version_base=None)

    # Remove the checkpoints from previous tests
    for f in glob.glob("data/test_checkpoints/test_training/*"):
        os.remove(f)


def post_config_setup(config):
    random.seed(config.habitat.seed)
    np.random.seed(config.habitat.seed)
    torch.manual_seed(config.habitat.seed)
    torch.cuda.manual_seed(config.habitat.seed)
    torch.backends.cudnn.deterministic = True
    if (
        config.habitat_baselines.force_torch_single_threaded
        and torch.cuda.is_available()
    ):
        torch.set_num_threads(1)


@pytest.mark.parametrize(
    "num_updates,expected_hash,add_opts",
    [
        (
            1,
            69515.362529343,
            [
                "habitat.task.measurements.predicate_task_success.sanity_end_task=True",
            ],
        ),
        (
            5,
            69498.58239580243,
            [
                "habitat_baselines.rl.policy.main_agent.hierarchical_policy.high_level_policy.debug_mode=True",
                "habitat.task.measurements.predicate_task_success.sanity_end_task=True",
                "habitat_baselines.rl.ppo.ppo_epoch=1",
            ],
        ),
        # Where the context length is greater than the rollout length.
        (
            10,
            69479.58418758967,
            [
                "habitat_baselines.rl.ppo.num_steps=5",
                "habitat_baselines.rl.policy.main_agent.hierarchical_policy.high_level_policy.debug_mode=True",
                "habitat_baselines.rl.ppo.ppo_epoch=1",
            ],
        ),
        # Extended training.
        (
            25,
            69448.26612490416,
            [
                "habitat.task.measurements.predicate_task_success.sanity_end_task=True",
            ],
        ),
    ],
)
def test_model_training(num_updates, expected_hash, add_opts):
    pre_config_setup()
    # Setup the training
    config = hydra.compose(
        "baseline/llarp.yaml",
        [
            f"habitat_baselines.num_updates={num_updates}",
            "habitat_baselines.num_checkpoints=-1",
            "habitat_baselines.checkpoint_interval=100000000",
            "habitat_baselines.total_num_steps=-1",
            "habitat_baselines.checkpoint_folder=data/test_checkpoints/test_training",
            "habitat.task.force_scene_per_worker=True",
            "habitat_baselines.num_environments=1",
            "habitat_baselines.writer_type=tb",
            "habitat_baselines.log_interval=1",
            "habitat_baselines.rl.ppo.num_mini_batch=1",
            "habitat.dataset.data_path=datasets/train_validation.pickle",
            "habitat_baselines.rl.policy.main_agent.hierarchical_policy.high_level_policy.llm_wrapper.llm_id=''",
            "habitat.simulator.agents.main_agent.joint_start_override=null",
            *add_opts,
        ],
    )

    with read_write(config):
        agent_config = get_agent_config(config.habitat.simulator)
        # Changing the visual observation size for speed
        for sim_sensor_config in agent_config.sim_sensors.values():
            sim_sensor_config.update({"height": 64, "width": 64})

    post_config_setup(config)

    trainer_init = baseline_registry.get_trainer(config.habitat_baselines.trainer_name)
    trainer = trainer_init(config)

    # Train
    trainer.train()
    train_hash = sum(
        v.sum().item() for v in trainer._agent.actor_critic.state_dict().values()
    )
    print(f"Got train hash {train_hash}")

    # With the new context window.
    assert train_hash == expected_hash


@pytest.mark.parametrize(
    "cfg_name,num_updates,expected_hash,add_opts,num_mini_batches",
    [
        ("baseline/llarp.yaml", 2, 69518.70178928264, [], 1),
        # Longer training.
        ("baseline/llarp.yaml", 10, 69522.72576177015, [], 1),
        # Longer training with minibatches.
        ("baseline/llarp.yaml", 10, 69478.0996894521, [], 4),
    ],
)
def test_model_training_batched(
    cfg_name, num_updates, expected_hash, add_opts, num_mini_batches
):
    pre_config_setup()
    # Setup the training
    config = hydra.compose(
        cfg_name,
        [
            f"habitat_baselines.num_updates={num_updates}",
            "habitat_baselines.num_checkpoints=-1",
            "habitat_baselines.checkpoint_interval=100000000",
            "habitat_baselines.total_num_steps=-1",
            "habitat_baselines.checkpoint_folder=data/test_checkpoints/test_training",
            "habitat.task.force_scene_per_worker=True",
            "habitat_baselines.rl.policy.main_agent.hierarchical_policy.high_level_policy.debug_mode=True",
            "habitat_baselines.num_environments=1",
            "habitat_baselines.writer_type=tb",
            "habitat_baselines.log_interval=1",
            f"habitat_baselines.rl.ppo.num_mini_batch={num_mini_batches}",
            "habitat.dataset.data_path=datasets/train_validation.pickle",
            "habitat_baselines.rl.policy.main_agent.hierarchical_policy.high_level_policy.llm_wrapper.llm_id=''",
            "habitat_baselines.vector_env_factory._target_='llarp.trainer.test_env_factory.TestVectorEnvFactory'",
            *add_opts,
        ],
    )

    post_config_setup(config)

    trainer_init = baseline_registry.get_trainer(config.habitat_baselines.trainer_name)
    trainer = trainer_init(config)

    # Train
    trainer.train()
    train_hash = sum(
        v.sum().item() for v in trainer._agent.actor_critic.state_dict().values()
    )
    print(f"Got train hash {train_hash}")

    # With the new context window.
    assert train_hash == expected_hash
