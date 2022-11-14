from __future__ import annotations

from typing import Any, Optional

class OptimizerWithMixedPrecision:
    def __init__(self, optimizer: Any, amp_lists: Any, use_pure_bf16: Any, use_bf16_guard: Any) -> None: ...
    def backward(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
        callbacks: Any | None = ...,
    ): ...
    def amp_init(
        self, place: Any, scope: Any | None = ..., test_program: Any | None = ..., use_bf16_test: bool = ...
    ) -> None: ...
    def apply_gradients(self, params_grads: Any): ...
    def apply_optimize(self, loss: Any, startup_program: Any, params_grads: Any): ...
    def minimize(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
    ): ...

def decorate_bf16(
    optimizer: Any, amp_lists: Any | None = ..., use_pure_bf16: bool = ..., use_bf16_guard: Any | None = ...
): ...
