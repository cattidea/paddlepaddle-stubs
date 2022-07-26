from __future__ import annotations

from sys import version as version
from typing import Any

from paddle.distributed.fleet.launch_utils import *

def get_cluster_from_args(args: Any, device_mode: Any, devices_per_proc: Any): ...
def cpuonly_check(args: Any): ...
def get_cluster_info(args: Any): ...
def get_global_envs(args: Any, tmp_dir: Any): ...
def launch_collective(args: Any) -> None: ...
def launch_ps(args: Any, distribute_mode: Any) -> None: ...
def infer_backend(args: Any) -> None: ...
def which_distributed_mode(args: Any): ...
def launch() -> None: ...
