from __future__ import annotations

from typing import Any, Optional

from paddle import fluid as fluid

from .parameter_server_optimizer import (
    ParameterServerOptimizer as ParameterServerOptimizer,
)

class ParameterServerGraphOptimizer(ParameterServerOptimizer):
    inner_opt: Any = ...
    meta_optimizers_white_list: Any = ...
    def __init__(self, optimizer: Any) -> None: ...
    def minimize(
        self,
        loss: Any,
        startup_program: Optional[Any] = ...,
        parameter_list: Optional[Any] = ...,
        no_grad_set: Optional[Any] = ...,
    ): ...
