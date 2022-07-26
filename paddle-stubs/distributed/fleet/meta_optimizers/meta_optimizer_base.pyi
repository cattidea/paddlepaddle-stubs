from __future__ import annotations

from typing import Any, Optional

from paddle.fluid.optimizer import Optimizer

class MetaOptimizerBase(Optimizer):
    inner_opt: Any = ...
    meta_optimizers_white_list: Any = ...
    meta_optimizers_black_list: Any = ...
    def __init__(self, optimizer: Any) -> None: ...
    def apply_gradients(self, params_grads: Any): ...
    def backward(
        self,
        loss: Any,
        startup_program: Optional[Any] = ...,
        parameter_list: Optional[Any] = ...,
        no_grad_set: Optional[Any] = ...,
        callbacks: Optional[Any] = ...,
    ): ...
    def apply_optimize(self, loss: Any, startup_program: Any, params_grads: Any): ...
    def minimize_impl(
        self,
        loss: Any,
        startup_program: Optional[Any] = ...,
        parameter_list: Optional[Any] = ...,
        no_grad_set: Optional[Any] = ...,
    ): ...
    def minimize(
        self,
        loss: Any,
        startup_program: Optional[Any] = ...,
        parameter_list: Optional[Any] = ...,
        no_grad_set: Optional[Any] = ...,
    ): ...
