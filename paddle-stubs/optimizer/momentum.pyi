from __future__ import annotations

from typing import Optional

from .._typing import Tensor
from ..fluid.clip import GradientClipBase
from ..fluid.regularizer import WeightDecayRegularizer
from .lr import LRScheduler
from .optimizer import Optimizer, ParameterConfig

class Momentum(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: float | LRScheduler = ...,
        momentum: float = ...,
        parameters: Optional[list[Tensor] | list[ParameterConfig]] = ...,
        use_nesterov: bool = ...,
        weight_decay: Optional[float | WeightDecayRegularizer] = ...,
        grad_clip: Optional[GradientClipBase] = ...,
        multi_precision: bool = ...,
        rescale_grad: float = ...,
        use_multi_tensor: bool = ...,
        name: Optional[str] = ...,
    ) -> None: ...
