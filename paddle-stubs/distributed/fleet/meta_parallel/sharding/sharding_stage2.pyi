from __future__ import annotations

from typing import Any, Optional

from paddle import nn

from ...meta_optimizers.dygraph_optimizer.sharding_optimizer_stage2 import (
    ShardingOptimizerStage2 as ShardingOptimizerStage2,
)
from ...utils.internal_storage import GradStorage as GradStorage
from .sharding_utils import Taskflow as Taskflow
from .sharding_utils import Type as Type

class ShardingStage2(nn.Layer):
    def __init__(
        self,
        layer: Any,
        sharding_optimizer: Any,
        group: Optional[Any] = ...,
        sync_buffers: bool = ...,
        buffer_max_size: Any = ...,
        auto_refresh_trainable: bool = ...,
        device: str = ...,
    ): ...
    def forward(self, *inputs: Any, **kwargs: Any): ...
    def set_state_dict(self, state_dict: Any, use_structured_name: bool = ...) -> None: ...
    def state_dict(
        self, destination: Optional[Any] = ..., include_sublayers: bool = ..., structured_name_prefix: str = ...
    ): ...
    def to(self, device: Optional[Any] = ..., dtype: Optional[Any] = ..., blocking: bool = ...) -> None: ...
    def __getattr__(self, name: Any): ...
