from __future__ import annotations

from typing import Any, Optional

class DecoupledWeightDecay:
    def __init__(self, coeff: float = ..., apply_decay_param_fun: Any | None = ..., **kwargs: Any) -> None: ...
    def backward(self, **kargs: Any): ...
    def apply_optimize(self, **kargs: Any): ...
    def minimize(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
    ): ...

def extend_with_decoupled_weight_decay(base_optimizer: Any): ...
