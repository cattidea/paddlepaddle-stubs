from __future__ import annotations

from typing import Any, Optional

from paddle.fluid.optimizer import Optimizer

class Momentum(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: Any,
        momentum: Any,
        parameter_list: Optional[Any] = ...,
        use_nesterov: bool = ...,
        regularization: Optional[Any] = ...,
        grad_clip: Optional[Any] = ...,
        multi_precision: bool = ...,
        rescale_grad: float = ...,
        name: Optional[str] = ...,
    ): ...
