from __future__ import annotations

from typing import Any, Optional

from paddle.fluid import core as core
from paddle.fluid.framework import Program as Program
from paddle.fluid.framework import default_main_program as default_main_program
from paddle.fluid.framework import default_startup_program as default_startup_program
from paddle.fluid.framework import device_guard as device_guard
from paddle.fluid.framework import name_scope as name_scope
from paddle.optimizer import Optimizer

class LookAhead(Optimizer):
    inner_optimizer: Any = ...
    alpha: Any = ...
    k: Any = ...
    type: str = ...
    helper: Any = ...
    def __init__(self, inner_optimizer: Any, alpha: float = ..., k: int = ..., name: Optional[Any] = ...) -> None: ...
    def step(self) -> None: ...
    def minimize(
        self,
        loss: Any,
        startup_program: Optional[Any] = ...,
        parameters: Optional[Any] = ...,
        no_grad_set: Optional[Any] = ...,
    ): ...
