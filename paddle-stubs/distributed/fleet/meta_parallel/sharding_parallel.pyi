from __future__ import annotations

from typing import Any

from paddle.base.dygraph.layers import Layer as Layer

from ..utils.hybrid_parallel_util import (
    broadcast_sharding_parameters as broadcast_sharding_parameters,
)
from ..utils.log_util import logger as logger
from .meta_parallel_base import MetaParallelBase as MetaParallelBase

class ShardingParallel(MetaParallelBase):
    def __init__(self, layers: Any, hcg: Any, **kwargs: Any) -> None: ...
