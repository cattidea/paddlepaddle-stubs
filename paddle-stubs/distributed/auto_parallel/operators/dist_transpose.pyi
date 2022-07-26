from __future__ import annotations

from typing import Any

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
from .common import (
    register_distributed_operator_impl as register_distributed_operator_impl,
)
from .common import (
    register_distributed_operator_impl_container as register_distributed_operator_impl_container,
)
from .dist_default import DistributedDefaultImpl0 as DistributedDefaultImpl0

class DistributedTranspose2(DistributedOperatorImplContainer):
    def __init__(self, op_type: Any) -> None: ...

class DistributedTranspose2Impl(DistributedOperatorImpl):
    def __init__(self, name: Any) -> None: ...
    def is_input_compatible(self, dist_op: Any): ...
    def is_output_compatible(self, dist_op: Any): ...
    def is_auto_compatible(self, dist_op: Any): ...
    def update_dims_mapping(self, dist_op: Any): ...
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> None: ...
    @staticmethod
    def backward(ctx: Any, *args: Any, **kwargs: Any) -> None: ...
