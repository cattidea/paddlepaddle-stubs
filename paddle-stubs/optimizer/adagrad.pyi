from __future__ import annotations

from typing import Any, Sequence

from .._typing import Tensor
from ..fluid.clip import GradientClipBase
from ..fluid.regularizer import WeightDecayRegularizer
from .lr import LRScheduler
from .optimizer import Optimizer, ParameterConfig

class Adagrad(Optimizer):
    type: str = ...
    initial_accumulator_value: Any = ...
    def __init__(
        self,
        learning_rate: float | LRScheduler,
        epsilon: float = ...,
        # TODO: Currently, pyright throws an error at below line.
        # parameters: Sequence[Tensor] | Sequence[ParameterConfig] | None = ...,
        parameters: Sequence[Tensor | ParameterConfig] | None = ...,
        weight_decay: float | WeightDecayRegularizer | None = ...,
        grad_clip: GradientClipBase | None = ...,
        name: str | None = ...,
        initial_accumulator_value: float = ...,
    ) -> None: ...
