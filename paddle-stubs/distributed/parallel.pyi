from __future__ import annotations

from typing import Any

from paddle.distributed import collective as collective

ParallelStrategy: Any

def init_parallel_env(): ...
def get_rank(): ...
def get_world_size(): ...
