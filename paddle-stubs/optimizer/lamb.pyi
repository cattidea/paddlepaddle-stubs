from __future__ import annotations

from collections.abc import Callable, Sequence

from .._typing import Tensor
from ..base.clip import GradientClipBase
from .lr import LRScheduler
from .optimizer import Optimizer, ParameterConfig

class Lamb(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: float | LRScheduler = ...,
        lamb_weight_decay: float = ...,
        beta1: float = ...,
        beta2: float = ...,
        epsilon: float = ...,
        # TODO: Currently, pyright throws an error at below line.
        # parameters: Sequence[Tensor] | Sequence[ParameterConfig] | None = ...,
        parameters: Sequence[Tensor | ParameterConfig] | None = ...,
        grad_clip: GradientClipBase | None = ...,
        exclude_from_weight_decay_fn: Callable[[Tensor], bool] | None = ...,
        name: str | None = ...,
    ) -> None: ...
