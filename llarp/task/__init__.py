#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os.path as osp

from habitat.gym.gym_definitions import _try_register

import llarp.config
from llarp.task.actions import *
from llarp.task.measures import *
from llarp.task.predicate_task import RearrangePredicateTask
from llarp.task.sensors import *


def easy_register(gym_id, local_path, overrides=None):
    if overrides is None:
        overrides = {}
    cfg_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
    full_path = osp.join(cfg_dir, "config", "task", local_path)
    _try_register(
        id_name=gym_id,
        entry_point="habitat.gym.gym_definitions:_make_habitat_gym_env",
        kwargs={"cfg_file_path": full_path, **overrides},
    )


easy_register("HabitatLanguageTask-v0", "lang_cond.yaml")
easy_register("HabitatVisLanguageTask-v0", "lang_cond.yaml", {"use_render_mode": True})
