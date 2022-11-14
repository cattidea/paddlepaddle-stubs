from __future__ import annotations

from typing import Optional, Sequence

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
        # TODO: Currently, pyright throws an error at below line.
        # parameters: Optional[Sequence[Tensor] | Sequence[ParameterConfig]] = ...,
        parameters: Optional[Sequence[Tensor | ParameterConfig]] = ...,
        weight_decay: Optional[float | WeightDecayRegularizer] = ...,
        grad_clip: Optional[GradientClipBase] = ...,
        name: Optional[str] = ...,
    ) -> None: ...
