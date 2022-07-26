from __future__ import annotations

from typing import Any, Optional

from ..base.private_helper_function import wait_server_ready as wait_server_ready
from .meta_optimizer_base import MetaOptimizerBase as MetaOptimizerBase

class GraphExecutionOptimizer(MetaOptimizerBase):
    inner_opt: Any = ...
    meta_optimizers_white_list: Any = ...
    meta_optimizers_black_list: Any = ...
    def __init__(self, optimizer: Any) -> None: ...
    def backward(
        self,
        loss: Any,
        startup_program: Optional[Any] = ...,
        parameter_list: Optional[Any] = ...,
        no_grad_set: Optional[Any] = ...,
        callbacks: Optional[Any] = ...,
    ) -> None: ...
    def minimize(
        self,
        loss: Any,
        startup_program: Optional[Any] = ...,
        parameter_list: Optional[Any] = ...,
        no_grad_set: Optional[Any] = ...,
    ): ...
