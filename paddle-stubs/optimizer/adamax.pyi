from __future__ import annotations

from collections.abc import Sequence

from .._typing import Tensor
from ..base.clip import GradientClipBase
from ..base.regularizer import WeightDecayRegularizer
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
        # parameters: Sequence[Tensor] | Sequence[AdamParameterConfig] | None = ...,
        parameters: Sequence[Tensor | AdamParameterConfig] | None = ...,
        weight_decay: float | WeightDecayRegularizer | None = ...,
        grad_clip: GradientClipBase | None = ...,
        name: str | None = ...,
    ) -> None: ...
