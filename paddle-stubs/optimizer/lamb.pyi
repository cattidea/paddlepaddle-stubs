from __future__ import annotations

from typing import Callable, Optional, Sequence

from .._typing import Tensor
from ..fluid.clip import GradientClipBase
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
        # parameters: Optional[Sequence[Tensor] | Sequence[ParameterConfig]] = ...,
        parameters: Optional[Sequence[Tensor | ParameterConfig]] = ...,
        grad_clip: Optional[GradientClipBase] = ...,
        exclude_from_weight_decay_fn: Optional[Callable[[Tensor], bool]] = ...,
        name: Optional[str] = ...,
    ) -> None: ...
