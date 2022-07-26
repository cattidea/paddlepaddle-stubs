from __future__ import annotations

from typing import Any

from paddle.fluid.dygraph.layers import Layer as Layer

from ..utils.hybrid_parallel_util import (
    broadcast_dp_parameters as broadcast_dp_parameters,
)
from ..utils.hybrid_parallel_util import broadcast_input_data as broadcast_input_data
from ..utils.hybrid_parallel_util import (
    broadcast_mp_parameters as broadcast_mp_parameters,
)
from ..utils.hybrid_parallel_util import (
    broadcast_sharding_parameters as broadcast_sharding_parameters,
)
from ..utils.log_util import logger as logger
from .meta_parallel_base import MetaParallelBase as MetaParallelBase

class TensorParallel(MetaParallelBase):
    def __init__(self, layers: Any, hcg: Any, **kwargs: Any) -> None: ...
