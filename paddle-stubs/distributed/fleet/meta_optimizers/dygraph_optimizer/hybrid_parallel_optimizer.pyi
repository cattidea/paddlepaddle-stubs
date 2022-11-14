from __future__ import annotations

from typing import Any, Optional

from paddle.fluid.framework import Variable as Variable
from paddle.optimizer import Optimizer as Optimizer

from ...base.topology import ParallelMode as ParallelMode
from ...utils.hybrid_parallel_util import (
    fused_allreduce_gradients as fused_allreduce_gradients,
)
from ...utils.hybrid_parallel_util import (
    sharding_reduce_gradients as sharding_reduce_gradients,
)
from ...utils.log_util import logger as logger

class HybridParallelClipGrad:
    def __init__(self, clip: Any, hcg: Any) -> None: ...
    def __getattr__(self, item: Any): ...
    def __call__(self, params_grads: Any): ...

class HybridParallelOptimizer:
    def __init__(self, optimizer: Any, hcg: Any, strategy: Any) -> None: ...
    def step(self) -> None: ...
    def minimize(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameters: Any | None = ...,
        no_grad_set: Any | None = ...,
    ): ...
    def __getattr__(self, item: Any): ...
