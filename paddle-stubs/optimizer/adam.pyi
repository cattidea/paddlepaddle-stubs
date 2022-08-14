from __future__ import annotations

from typing import Optional

from typing_extensions import NotRequired

from .._typing import Tensor
from ..fluid.clip import GradientClipBase
from ..fluid.regularizer import WeightDecayRegularizer
from .lr import LRScheduler
from .optimizer import Optimizer, ParameterConfig

class AdamParameterConfig(ParameterConfig):
    beta1: NotRequired[float | Tensor]
    beta2: NotRequired[float | Tensor]

class Adam(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: float | LRScheduler = ...,
        beta1: float | Tensor = ...,
        beta2: float | Tensor = ...,
        epsilon: float = ...,
        parameters: Optional[list[Tensor] | list[AdamParameterConfig]] = ...,
        weight_decay: Optional[float | WeightDecayRegularizer] = ...,
        grad_clip: Optional[GradientClipBase] = ...,
        lazy_mode: bool = ...,
        multi_precision: bool = ...,
        use_multi_tensor: bool = ...,
        name: Optional[str] = ...,
    ) -> None: ...
