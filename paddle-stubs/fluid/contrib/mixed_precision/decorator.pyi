from __future__ import annotations

from typing import Any, Optional

class OptimizerWithMixedPrecision:
    def __init__(
        self,
        optimizer: Any,
        amp_lists: Any,
        init_loss_scaling: Any,
        use_dynamic_loss_scaling: Any,
        incr_every_n_steps: Any,
        decr_every_n_nan_or_inf: Any,
        incr_ratio: Any,
        decr_ratio: Any,
        use_pure_fp16: Any,
        use_fp16_guard: Any,
    ) -> None: ...
    def get_loss_scaling(self): ...
    def get_scaled_loss(self): ...
    def backward(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
        callbacks: Any | None = ...,
    ): ...
    def amp_init(
        self, place: Any, scope: Any | None = ..., test_program: Any | None = ..., use_fp16_test: bool = ...
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

def decorate(
    optimizer: Any,
    amp_lists: Any | None = ...,
    init_loss_scaling: Any = ...,
    incr_every_n_steps: int = ...,
    decr_every_n_nan_or_inf: int = ...,
    incr_ratio: float = ...,
    decr_ratio: float = ...,
    use_dynamic_loss_scaling: bool = ...,
    use_pure_fp16: bool = ...,
    use_fp16_guard: Any | None = ...,
): ...
