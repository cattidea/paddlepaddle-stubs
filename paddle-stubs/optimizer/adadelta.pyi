from __future__ import annotations

from typing import Optional

from .._typing import Tensor
from ..fluid.clip import GradientClipBase
from ..fluid.regularizer import WeightDecayRegularizer
from .lr import LRScheduler
from .optimizer import Optimizer, ParameterConfig

class Adadelta(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: float | LRScheduler = ...,
        epsilon: float = ...,
        rho: float = ...,
        parameters: Optional[list[Tensor] | list[ParameterConfig]] = ...,
        weight_decay: Optional[float | WeightDecayRegularizer] = ...,
        grad_clip: Optional[GradientClipBase] = ...,
        name: Optional[str] = ...,
    ) -> None: ...
