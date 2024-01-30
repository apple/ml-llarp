#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""
Script to launch Habitat Baselines trainer.
"""

import os
import os.path as osp
import random

import gym
import hydra
import numpy as np
import torch
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import register_hydra_plugin
from habitat_baselines.config.default_structured_configs import \
    HabitatBaselinesConfigPlugin

import llarp.policies
import llarp.task
import llarp.trainer
from llarp.config import default_structured_configs

# Suppress gym import warnings.
gym.logger.set_level(40)


def on_save_ckpt_callback(save_file_path: str) -> None:
    """
    Process the saved checkpoints here if desired.
    """

    pass


@hydra.main(
    version_base=None,
    config_path="config",
    # The default is overridden in the launch command.
    config_name="pointnav/ppo_pointnav_example",
)
def main(cfg):
    cfg = patch_config(cfg)
    random.seed(cfg.habitat.seed)
    np.random.seed(cfg.habitat.seed)
    torch.manual_seed(cfg.habitat.seed)

    if cfg.habitat_baselines.force_torch_single_threaded and torch.cuda.is_available():
        torch.set_num_threads(1)

    from habitat_baselines.common.baseline_registry import baseline_registry

    trainer_init = baseline_registry.get_trainer(cfg.habitat_baselines.trainer_name)
    trainer = trainer_init(cfg)

    if cfg.habitat_baselines.evaluate:
        trainer.eval()
    else:
        trainer.train()


if __name__ == "__main__":
    register_hydra_plugin(HabitatBaselinesConfigPlugin)
    main()
