from __future__ import annotations

from typing import Any, Optional

from ..fluid import core as core
from ..fluid import framework as framework
from ..fluid.framework import Variable as Variable
from .optimizer import Optimizer as Optimizer

class RMSProp(Optimizer):
    type: str = ...
    def __init__(
        self,
        learning_rate: Any,
        rho: float = ...,
        epsilon: float = ...,
        momentum: float = ...,
        centered: bool = ...,
        parameters: Optional[Any] = ...,
        weight_decay: Optional[Any] = ...,
        grad_clip: Optional[Any] = ...,
        name: Optional[Any] = ...,
    ) -> None: ...
