from __future__ import annotations

from typing import Any, Optional

from paddle.base.framework import Variable as Variable

from .dist_attribute import OperatorDistributedAttribute as OperatorDistributedAttribute
from .dist_attribute import TensorDistributedAttribute as TensorDistributedAttribute
from .dist_context import (
    get_default_distributed_context as get_default_distributed_context,
)
from .dist_op import DistributedModule as DistributedModule
from .dist_tensor import DistributedTensor as DistributedTensor

def shard_tensor(x: Any, dist_attr: Any | None = ...): ...
def shard_op(op_fn: Any, dist_attr: Any | None = ...): ...
