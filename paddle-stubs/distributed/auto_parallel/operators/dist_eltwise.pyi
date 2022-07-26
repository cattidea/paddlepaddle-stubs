from __future__ import annotations

from typing import Any

from paddle.distributed.fleet.meta_optimizers.common import OP_ROLE_KEY as OP_ROLE_KEY
from paddle.distributed.fleet.meta_optimizers.common import (
    OP_ROLE_VAR_KEY as OP_ROLE_VAR_KEY,
)
from paddle.distributed.fleet.meta_optimizers.common import OpRole as OpRole
from paddle.fluid import core as core
from paddle.fluid import unique_name as unique_name
from paddle.fluid.data_feeder import check_dtype as check_dtype
from paddle.fluid.data_feeder import (
    check_variable_and_dtype as check_variable_and_dtype,
)
from paddle.fluid.framework import Parameter as Parameter
from paddle.fluid.framework import Program as Program
from paddle.fluid.framework import Variable as Variable
from paddle.fluid.framework import program_guard as program_guard

from ..dist_attribute import (
    OperatorDistributedAttribute as OperatorDistributedAttribute,
)
from ..process_group import new_process_group as new_process_group
from ..utils import (
    compute_compatible_and_update_dim_mapping as compute_compatible_and_update_dim_mapping,
)
from ..utils import compute_compatible_dim_mapping as compute_compatible_dim_mapping
from ..utils import compute_compatible_dims_mapping as compute_compatible_dims_mapping
from ..utils import is_dim_replicate as is_dim_replicate
from ..utils import is_dim_shard as is_dim_shard
from ..utils import is_valid_list_index as is_valid_list_index
from .common import DistributedOperatorImpl as DistributedOperatorImpl
from .common import DistributedOperatorImplContainer as DistributedOperatorImplContainer
from .common import is_elementwise_op as is_elementwise_op
from .common import (
    register_distributed_operator_impl as register_distributed_operator_impl,
)
from .common import (
    register_distributed_operator_impl_container as register_distributed_operator_impl_container,
)
from .dist_default import DistributedDefaultImpl0 as DistributedDefaultImpl0

class DistributedElementwise(DistributedOperatorImplContainer):
    def __init__(self, op_type: Any) -> None: ...

class DistributedElementwiseImpl0(DistributedOperatorImpl):
    def __init__(self, name: Any) -> None: ...
    def is_input_compatible(self, dist_op: Any): ...
    def is_output_compatible(self, dist_op: Any): ...
    def is_auto_compatible(self, dist_op: Any): ...
    def update_dims_mapping(self, dist_op: Any): ...
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> None: ...
    @staticmethod
    def backward(ctx: Any, *args: Any, **kwargs: Any) -> None: ...
