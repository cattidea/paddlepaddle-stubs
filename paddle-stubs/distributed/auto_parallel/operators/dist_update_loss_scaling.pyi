from __future__ import annotations

from typing import Any

from ..utils import set_dist_op_desc_original_id as set_dist_op_desc_original_id
from .common import DistributedOperatorImpl as DistributedOperatorImpl
from .common import DistributedOperatorImplContainer as DistributedOperatorImplContainer
from .common import (
    register_distributed_operator_impl as register_distributed_operator_impl,
)
from .common import (
    register_distributed_operator_impl_container as register_distributed_operator_impl_container,
)

class DistributedUpdateLossScaling(DistributedOperatorImplContainer):
    def __init__(self, op_type: Any) -> None: ...

class DistributedUpdateLossScalingImpl(DistributedOperatorImpl):
    def __init__(self, name: Any) -> None: ...
    def is_input_compatible(self, dist_op: Any) -> None: ...
    def is_output_compatible(self, dist_op: Any) -> None: ...
    def is_auto_compatible(self, dist_op: Any) -> None: ...
    def update_dims_mapping(self, dist_op: Any) -> None: ...
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> None: ...
    @staticmethod
    def backward(ctx: Any, *args: Any, **kwargs: Any) -> None: ...
