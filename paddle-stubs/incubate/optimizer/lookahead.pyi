from __future__ import annotations

from typing import Any, Optional

from paddle.base import core as core
from paddle.base.framework import Program as Program
from paddle.base.framework import default_main_program as default_main_program
from paddle.base.framework import default_startup_program as default_startup_program
from paddle.base.framework import device_guard as device_guard
from paddle.base.framework import name_scope as name_scope
from paddle.optimizer import Optimizer

class LookAhead(Optimizer):
    inner_optimizer: Any = ...
    alpha: Any = ...
    k: Any = ...
    type: str = ...
    helper: Any = ...
    def __init__(self, inner_optimizer: Any, alpha: float = ..., k: int = ..., name: str | None = ...) -> None: ...
    def step(self) -> None: ...
    def minimize(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameters: Any | None = ...,
        no_grad_set: Any | None = ...,
    ): ...
