from __future__ import annotations

from typing import Optional, Sequence

from .._typing import Tensor
from ..fluid.clip import GradientClipBase
from ..fluid.regularizer import WeightDecayRegularizer
from .adam import AdamParameterConfig
from .lr import LRScheduler
from .optimizer import Optimizer

class Adamax(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: float | LRScheduler = ...,
        beta1: float = ...,
        beta2: float = ...,
        epsilon: float = ...,
        # TODO: Currently, pyright throws an error at below line.
        # parameters: Optional[Sequence[Tensor] | Sequence[AdamParameterConfig]] = ...,
        parameters: Optional[Sequence[Tensor | AdamParameterConfig]] = ...,
        weight_decay: Optional[float | WeightDecayRegularizer] = ...,
        grad_clip: Optional[GradientClipBase] = ...,
        name: Optional[str] = ...,
    ) -> None: ...
