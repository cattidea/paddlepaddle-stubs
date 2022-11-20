from __future__ import annotations

from collections.abc import Callable, Sequence

from .._typing import Tensor
from ..fluid.clip import GradientClipBase
from ..fluid.regularizer import WeightDecayRegularizer
from .adam import Adam, AdamParameterConfig
from .lr import LRScheduler

class AdamW(Adam):
    type: str = ...
    def __init__(
        self,
        learning_rate: float | LRScheduler = ...,
        beta1: float | Tensor = ...,
        beta2: float | Tensor = ...,
        epsilon: float = ...,
        # TODO: Currently, pyright throws an error at below line.
        # parameters: Sequence[Tensor] | Sequence[AdamParameterConfig] | None = ...,
        parameters: Sequence[Tensor | AdamParameterConfig] | None = ...,
        weight_decay: float | WeightDecayRegularizer | None = ...,
        lr_ratio: Callable[[Tensor], float] | None = ...,
        apply_decay_param_fun: Callable[[str], bool] | None = ...,
        grad_clip: GradientClipBase | None = ...,
        lazy_mode: bool = ...,
        multi_precision: bool = ...,
        name: str | None = ...,
    ) -> None: ...
