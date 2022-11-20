from __future__ import annotations

from collections.abc import Sequence

from .._typing import Tensor
from ..fluid.clip import GradientClipBase
from ..fluid.regularizer import WeightDecayRegularizer
from .lr import LRScheduler
from .optimizer import Optimizer, ParameterConfig

class RMSProp(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: float | LRScheduler = ...,
        rho: float = ...,
        epsilon: float = ...,
        momentum: float = ...,
        centered: bool = ...,
        # TODO: Currently, pyright throws an error at below line.
        # parameters: Sequence[Tensor] | Sequence[ParameterConfig] | None = ...,
        parameters: Sequence[Tensor | ParameterConfig] | None = ...,
        weight_decay: float | WeightDecayRegularizer | None = ...,
        grad_clip: GradientClipBase | None = ...,
        name: str | None = ...,
    ) -> None: ...
