from __future__ import annotations

from typing import Any, Optional

from paddle.base.optimizer import Optimizer

class Momentum(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: Any,
        momentum: Any,
        parameter_list: Any | None = ...,
        use_nesterov: bool = ...,
        regularization: Any | None = ...,
        grad_clip: Any | None = ...,
        multi_precision: bool = ...,
        rescale_grad: float = ...,
        name: str | None = ...,
    ): ...
