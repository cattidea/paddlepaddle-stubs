from __future__ import annotations

from typing import Any

from paddle.base import core as core
from paddle.base import unique_name as unique_name
from paddle.base.data_feeder import check_dtype as check_dtype
from paddle.base.data_feeder import (
    check_variable_and_dtype as check_variable_and_dtype,
)
from paddle.base.framework import Parameter as Parameter
from paddle.base.framework import Program as Program
from paddle.base.framework import Variable as Variable
from paddle.base.framework import program_guard as program_guard

from ..utils import (
    compute_compatible_and_update_dim_mapping as compute_compatible_and_update_dim_mapping,
)
from ..utils import compute_compatible_dim_mapping as compute_compatible_dim_mapping
from ..utils import compute_compatible_dims_mapping as compute_compatible_dims_mapping
from ..utils import is_dim_replicate as is_dim_replicate
from ..utils import is_dim_shard as is_dim_shard
from ..utils import is_valid_list_index as is_valid_list_index
from ..utils import set_dist_op_desc_original_id as set_dist_op_desc_original_id
from .common import DistributedOperatorImpl as DistributedOperatorImpl
from .common import DistributedOperatorImplContainer as DistributedOperatorImplContainer
from .common import (
    register_distributed_operator_impl as register_distributed_operator_impl,
)
from .common import (
    register_distributed_operator_impl_container as register_distributed_operator_impl_container,
)
from .dist_default import DistributedDefaultImpl0 as DistributedDefaultImpl0

class DistributedFillConstantBatchSizeLike(DistributedOperatorImplContainer):
    def __init__(self, op_type: Any) -> None: ...

class DistributedFillConstantBatchSizeLikeImpl0(DistributedOperatorImpl):
    def __init__(self, name: Any) -> None: ...
    def is_input_compatible(self, dist_op: Any): ...
    def is_output_compatible(self, dist_op: Any): ...
    def is_auto_compatible(self, dist_op: Any): ...
    def update_dims_mapping(self, dist_op: Any): ...
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> None: ...
    @staticmethod
    def backward(ctx: Any, *args: Any, **kwargs: Any) -> None: ...
