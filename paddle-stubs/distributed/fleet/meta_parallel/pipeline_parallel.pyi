from __future__ import annotations

from typing import Any, Optional

from ..meta_optimizers.dygraph_optimizer import (
    HybridParallelGradScaler as HybridParallelGradScaler,
)
from ..meta_optimizers.dygraph_optimizer import (
    HybridParallelOptimizer as HybridParallelOptimizer,
)
from ..utils.hybrid_parallel_util import (
    broadcast_dp_parameters as broadcast_dp_parameters,
)
from ..utils.hybrid_parallel_util import (
    broadcast_mp_parameters as broadcast_mp_parameters,
)
from ..utils.hybrid_parallel_util import (
    broadcast_sharding_parameters as broadcast_sharding_parameters,
)
from ..utils.log_util import logger as logger
from .meta_parallel_base import MetaParallelBase as MetaParallelBase
from .parallel_layers.pp_layers import PipelineLayer as PipelineLayer
from .pp_utils.utils import is_float_tensor as is_float_tensor

class PipelineParallel(MetaParallelBase):
    use_data_parallel: Any = ...
    use_model_parallel: Any = ...
    use_sharding_parallel: Any = ...
    total_loss: Any = ...
    micro_batch_size: Any = ...
    accumulate_steps: Any = ...
    num_stages: Any = ...
    stage_id: Any = ...
    pp_group: Any = ...
    is_first_stage: Any = ...
    is_last_stage: Any = ...
    global_rank: Any = ...
    micro_batch_id: int = ...
    def __init__(self, layers: Any, hcg: Any, strategy: Any) -> None: ...
    scaler: Any = ...
    data: Any = ...
    def forward_backward_pipeline(self, data: Any, scaler: Any | None = ...): ...
    optimizer: Any = ...
    lr_scheduler: Any = ...
    def train_batch(self, data: Any, optimizer: Any, lr_scheduler: Any | None = ..., scaler: Any | None = ...): ...
    train_loss: Any = ...
    def eval_batch(self, data: Any, compute_loss: bool = ...): ...
