from __future__ import annotations

from typing import Any, Optional

from paddle.base.initializer import Constant as Constant
from paddle.optimizer import Optimizer

class DistributedFusedLamb(Optimizer):
    helper: Any = ...
    def __init__(
        self,
        learning_rate: float = ...,
        lamb_weight_decay: float = ...,
        beta1: float = ...,
        beta2: float = ...,
        epsilon: float = ...,
        parameters: Any | None = ...,
        grad_clip: Any | None = ...,
        exclude_from_weight_decay_fn: Any | None = ...,
        clip_after_allreduce: bool = ...,
        is_grad_scaled_by_nranks: bool = ...,
        alignment: int = ...,
        use_master_param_norm: bool = ...,
        name: str | None = ...,
    ) -> None: ...
    def apply_optimize(self, params_grads: Any) -> None: ...
    def apply_gradients(self, params_grads: Any) -> None: ...
