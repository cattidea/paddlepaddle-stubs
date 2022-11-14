from __future__ import annotations

from typing import Optional, Sequence

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
        # parameters: Optional[Sequence[Tensor] | Sequence[ParameterConfig]] = ...,
        parameters: Optional[Sequence[Tensor | ParameterConfig]] = ...,
        use_nesterov: bool = ...,
        weight_decay: Optional[float | WeightDecayRegularizer] = ...,
        grad_clip: Optional[GradientClipBase] = ...,
        multi_precision: bool = ...,
        rescale_grad: float = ...,
        use_multi_tensor: bool = ...,
        name: Optional[str] = ...,
    ) -> None: ...
