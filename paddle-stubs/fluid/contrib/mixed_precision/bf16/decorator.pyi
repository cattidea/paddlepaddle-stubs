from __future__ import annotations

from typing import Any, Optional

class OptimizerWithMixedPrecision:
    def __init__(self, optimizer: Any, amp_lists: Any, use_pure_bf16: Any, use_bf16_guard: Any) -> None: ...
    def backward(
        self,
        loss: Any,
        startup_program: Optional[Any] = ...,
        parameter_list: Optional[Any] = ...,
        no_grad_set: Optional[Any] = ...,
        callbacks: Optional[Any] = ...,
    ): ...
    def amp_init(
        self, place: Any, scope: Optional[Any] = ..., test_program: Optional[Any] = ..., use_bf16_test: bool = ...
    ) -> None: ...
    def apply_gradients(self, params_grads: Any): ...
    def apply_optimize(self, loss: Any, startup_program: Any, params_grads: Any): ...
    def minimize(
        self,
        loss: Any,
        startup_program: Optional[Any] = ...,
        parameter_list: Optional[Any] = ...,
        no_grad_set: Optional[Any] = ...,
    ): ...

def decorate_bf16(
    optimizer: Any, amp_lists: Optional[Any] = ..., use_pure_bf16: bool = ..., use_bf16_guard: Optional[Any] = ...
): ...
