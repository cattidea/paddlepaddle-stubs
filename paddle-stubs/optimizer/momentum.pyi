from __future__ import annotations

from collections.abc import Sequence

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
        # TODO: Currently, pyright throws an error at below line.
        # parameters: Sequence[Tensor] | Sequence[ParameterConfig] | None = ...,
        parameters: Sequence[Tensor | ParameterConfig] | None = ...,
        use_nesterov: bool = ...,
        weight_decay: float | WeightDecayRegularizer | None = ...,
        grad_clip: GradientClipBase | None = ...,
        multi_precision: bool = ...,
        rescale_grad: float = ...,
        use_multi_tensor: bool = ...,
        name: str | None = ...,
    ) -> None: ...
