#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, List, Optional, Type

import torch
from habitat import ThreadedVectorEnv, VectorEnv, logger, make_dataset
from habitat.config import read_write
from habitat.gym import make_gym_from_config
from habitat_baselines.common.env_factory import VectorEnvFactory

from llarp.dataset.episodes import LangRearrangeDatasetV0, LangRearrangeEpisode

if TYPE_CHECKING:
    from omegaconf import DictConfig


def split_scenes(scenes, n_envs):
    # Mapping to env to scenes allowed in this env.
    scene_splits: List[List[str]] = [[] for _ in range(n_envs)]
    n_scenes = len(scenes)

    # Map all the scenes into an env
    for scene_i, scene in enumerate(scenes):
        scene_splits[scene_i % n_envs].append(scene)

    # Handle left over envs that didn't get any scenes.
    for env_i in range(n_scenes, n_envs):
        scene_splits[env_i].append(scenes[env_i % n_scenes])
    return scene_splits


def habitat_create_dataset(
    config,
    is_first_rank,
    num_environments,
    force_scene_per_worker: bool,
    filter_down_num: Optional[int],
    filter_instructs: Optional[List[str]] = None,
):
    configs = []
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    # Limit episodes
    if filter_down_num is not None:
        dataset.episodes = dataset.episodes[:filter_down_num]

    scenes = dataset.scene_ids

    scene_splits = split_scenes(scenes, num_environments)

    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()

    if force_scene_per_worker:
        for i in range(num_environments):
            n_scenes_in_split = len(scene_splits[i])
            # Offset which scene we are taking by the rank.
            scene_splits[i] = [scene_splits[i][rank % n_scenes_in_split]]

    all_env_episodes = defaultdict(list)
    ignored_count = 0
    worker_to_ep_counts = defaultdict(lambda: defaultdict(int))

    for ep in dataset.episodes:
        match_env_idx = None
        for i, scene_split in enumerate(scene_splits):
            if ep.scene_id in scene_split:
                match_env_idx = i
                break
        if match_env_idx is None:
            ignored_count += 1
            continue
        all_env_episodes[match_env_idx].append(ep)
        if isinstance(ep, LangRearrangeEpisode):
            worker_to_ep_counts[match_env_idx][ep.instruct_id] += 1

    if force_scene_per_worker:
        n_distinct = len(set(x for scene_split in scene_splits for x in scene_split))
        print(f"Worker using {n_distinct}/{len(set(scenes))} scenes")
        # Ensure the objects match per scene.
        for env_idx in range(num_environments):
            objs = None
            for ep in all_env_episodes[env_idx]:
                ep_objs = [x[0] for x in ep.rigid_objs]
                if objs is None:
                    objs = ep_objs
                elif ep_objs != objs:
                    raise ValueError("Objects are different within a scene!")
            for k, v in worker_to_ep_counts[env_idx].items():
                print(f" - {env_idx}, {k}: {v}")

    env_datasets = []
    unfound_idxs = []
    found_idxs = []
    for env_index in range(num_environments):
        print(f"Rank {rank}, Env {env_index}: {scene_splits[env_index]}")
        proc_config = config.copy()
        with read_write(proc_config):
            task_config = proc_config.habitat
            task_config.seed = task_config.seed + env_index
            remove_measure_names = []
            if not is_first_rank:
                # Filter out non rank0_measure from the task config if we are not on rank0.
                remove_measure_names.extend(task_config.task.rank0_measure_names)
            if (env_index != 0) or not is_first_rank:
                # Filter out non-rank0_env0 measures from the task config if we
                # are not on rank0 env0.
                remove_measure_names.extend(task_config.task.rank0_env0_measure_names)

            task_config.task.measurements = {
                k: v
                for k, v in task_config.task.measurements.items()
                if k not in remove_measure_names
            }

        configs.append(proc_config)

        env_episodes = all_env_episodes[env_index]
        if len(env_episodes) == 0:
            assert env_index != 0
            env_episodes = all_env_episodes[env_index - 1]
            # raise ValueError("Found no initial episodes for scene")

        # Potentially filter episodes.
        if filter_instructs is not None:
            env_episodes = [
                ep for ep in env_episodes if ep.instruct_id in filter_instructs
            ]
        if len(env_episodes) == 0:
            # Filtered to no episodes for this scene. Just grab the episodes from the previous worker.
            unfound_idxs.append(env_index)
        else:
            found_idxs.append(env_index)

        env_datasets.append(
            LangRearrangeDatasetV0(
                config=config.habitat.dataset, preset_eps=env_episodes
            )
        )

    for env_index in unfound_idxs:
        found_env_index = found_idxs[env_index % len(found_idxs)]
        env_datasets[env_index].episodes = env_datasets[found_env_index].episodes
    return env_datasets, configs


class CustomVectorEnvFactory(VectorEnvFactory):
    def __init__(self):
        pass

    def construct_envs(
        self,
        config: "DictConfig",
        workers_ignore_signals: bool = False,
        enforce_scenes_greater_eq_environments: bool = False,
        is_first_rank: bool = True,
    ) -> VectorEnv:
        env_datasets, configs = habitat_create_dataset(
            config,
            is_first_rank,
            config.habitat_baselines.num_environments,
            config.habitat.task.force_scene_per_worker,
            config.habitat.task.filter_down_num,
            config.habitat.task.filter_instructs,
        )

        if int(os.environ.get("HABITAT_ENV_DEBUG", 0)):
            logger.warn(
                "Using the debug Vector environment interface. Expect slower performance."
            )
            vector_env_cls = ThreadedVectorEnv
        else:
            vector_env_cls = VectorEnv

        envs = vector_env_cls(
            make_env_fn=make_gym_from_config,
            env_fn_args=tuple(
                (c, env_dataset) for c, env_dataset in zip(configs, env_datasets)
            ),
            workers_ignore_signals=workers_ignore_signals,
        )

        if config.habitat.simulator.renderer.enable_batch_renderer:
            envs.initialize_batch_renderer(config)

        return envs
