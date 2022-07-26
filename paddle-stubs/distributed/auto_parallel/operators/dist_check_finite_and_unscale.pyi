from __future__ import annotations

from typing import Any

from paddle.distributed.fleet.meta_optimizers.common import (
    OP_ROLE_VAR_KEY as OP_ROLE_VAR_KEY,
)
from paddle.fluid import unique_name as unique_name
from paddle.fluid.data_feeder import check_dtype as check_dtype
from paddle.fluid.data_feeder import (
    check_variable_and_dtype as check_variable_and_dtype,
)

from ..dist_attribute import (
    OperatorDistributedAttribute as OperatorDistributedAttribute,
)
from ..process_group import new_process_group as new_process_group
from ..utils import set_dist_op_desc_original_id as set_dist_op_desc_original_id
from ..utils import set_var_dist_attr as set_var_dist_attr
from .common import DistributedOperatorImpl as DistributedOperatorImpl
from .common import DistributedOperatorImplContainer as DistributedOperatorImplContainer
from .common import (
    register_distributed_operator_impl as register_distributed_operator_impl,
)
from .common import (
    register_distributed_operator_impl_container as register_distributed_operator_impl_container,
)

world_process_group: Any

class DistributedCheckFiniteAndUnscale(DistributedOperatorImplContainer):
    def __init__(self, op_type: Any) -> None: ...

class DistributedCheckFiniteAndUnscaleImpl(DistributedOperatorImpl):
    def __init__(self, name: Any) -> None: ...
    def is_input_compatible(self, dist_op: Any) -> None: ...
    def is_output_compatible(self, dist_op: Any) -> None: ...
    def is_auto_compatible(self, dist_op: Any) -> None: ...
    def update_dims_mapping(self, dist_op: Any) -> None: ...
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> None: ...
    @staticmethod
    def backward(ctx: Any, *args: Any, **kwargs: Any) -> None: ...
