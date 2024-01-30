#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import contextlib
import functools
import io
import os
import pickle
import signal
import socket
import subprocess
import threading
from os import path as osp
from typing import (Any, Callable, List, Optional, Tuple, TypeVar, Union,
                    overload)

import ifcfg
import numpy as np
import torch
from habitat import logger
from habitat_baselines.rl.ddppo.ddp_utils import (DEFAULT_MAIN_ADDR,
                                                  DEFAULT_PORT,
                                                  DEFAULT_PORT_RANGE,
                                                  SLURM_JOBID,
                                                  get_distrib_size, get_ifname)
from omegaconf import DictConfig
from torch import distributed as distrib


def is_multi_node():
    return int(os.environ.get("NUM_NODES", 1)) > 1


def get_main_addr() -> str:
    if is_multi_node():
        k = "MASTER_ADDR"
    else:
        k = "MAIN_ADDR"
    return os.environ.get(k, DEFAULT_MAIN_ADDR)


def init_distrib_slurm(
    backend: str = "nccl",
) -> Tuple[int, torch.distributed.TCPStore]:  # type: ignore
    assert torch.distributed.is_available(), "torch.distributed must be available"

    if "GLOO_SOCKET_IFNAME" not in os.environ:
        os.environ["GLOO_SOCKET_IFNAME"] = get_ifname()

    if "NCCL_SOCKET_IFNAME" not in os.environ:
        os.environ["NCCL_SOCKET_IFNAME"] = get_ifname()

    local_rank, world_rank, world_size = get_distrib_size()

    if is_multi_node():
        k = "MASTER_PORT"
    else:
        k = "MAIN_PORT"
    main_port = int(os.environ.get(k, DEFAULT_PORT))
    if SLURM_JOBID is not None:
        main_port += int(SLURM_JOBID) % int(
            os.environ.get("MAIN_PORT_RANGE", DEFAULT_PORT_RANGE)
        )
    main_addr = get_main_addr()

    print(
        f"Setting up TCP store with {main_addr}, {main_port}, {world_size}, {world_rank} on backend {backend}"
    )
    tcp_store = distrib.TCPStore(  # type: ignore
        main_addr, main_port, world_size, world_rank == 0
    )
    distrib.init_process_group(
        backend, store=tcp_store, rank=world_rank, world_size=world_size
    )

    return local_rank, tcp_store
