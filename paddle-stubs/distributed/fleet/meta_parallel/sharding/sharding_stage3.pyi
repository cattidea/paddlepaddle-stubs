from __future__ import annotations

from collections import deque as deque
from itertools import chain as chain
from typing import Any, Optional

from paddle import nn
from paddle.autograd import PyLayer

from ...utils.internal_storage import GradStorage as GradStorage
from .sharding_utils import ShardingClipGrad as ShardingClipGrad
from .sharding_utils import Type as Type
from .sharding_utils import device_guard as device_guard

alignment: Any
align: Any
CHECK_LAYER: Any

class ShardingStage3(nn.Layer):
    def __init__(
        self,
        layer: Any,
        optimizer: Any,
        group: Optional[Any] = ...,
        sync_buffers: bool = ...,
        device: str = ...,
        segment_size: Any = ...,
        pertrain_sync_models: bool = ...,
        offload: bool = ...,
        sync_comm: bool = ...,
    ) -> None: ...
    def forward(self, *inputs: Any, **kwargs: Any): ...
    def set_state_dict(self, state_dict: Any, use_structured_name: bool = ...) -> None: ...
    def state_dict(
        self, destination: Optional[Any] = ..., include_sublayers: bool = ..., structured_name_prefix: str = ...
    ): ...
    def __getattr__(self, name: Any): ...
    def get_all_parameters(self, convert2cpu: bool = ...): ...

def ForwardPreHooks(
    layer: Any,
    order_tracer: Any,
    trainable_params: Any,
    param2buffer: Any,
    rank: Any,
    group: Any,
    sync_comm: Any,
    offload: Any,
    task_flow: Any,
) -> None: ...

class ForwardPostHooks(PyLayer):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Any,
        layer: Any,
        order_tracer: Any,
        trainable_params: Any,
        param2buffer: Any,
        param2buffer_size: Any,
        rank: Any,
        group: Any,
        sync_comm: Any,
        offload: Any,
        task_flow: Any,
    ): ...
    @staticmethod
    def backward(ctx: Any, *args: Any): ...

class TaskFlow:
    full_param: Any = ...
    full_grad: Any = ...
    use_calc: Any = ...
    callback: Any = ...
    def __init__(
        self, full_param: Any = ..., full_grad: Any = ..., use_calc: Any = ..., callback: Optional[Any] = ...
    ) -> None: ...
