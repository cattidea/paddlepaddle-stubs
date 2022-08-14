from __future__ import annotations

from typing import Callable, Optional

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
        parameters: Optional[list[Tensor] | list[AdamParameterConfig]] = ...,
        weight_decay: Optional[float | WeightDecayRegularizer] = ...,
        lr_ratio: Optional[Callable[[Tensor], float]] = ...,
        apply_decay_param_fun: Optional[Callable[[str], bool]] = ...,
        grad_clip: Optional[GradientClipBase] = ...,
        lazy_mode: bool = ...,
        multi_precision: bool = ...,
        name: Optional[str] = ...,
    ) -> None: ...
