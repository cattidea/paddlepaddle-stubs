from __future__ import annotations

from typing import Any, Optional

from .line_search import strong_wolfe as strong_wolfe
from .utils import (
    check_initial_inverse_hessian_estimate as check_initial_inverse_hessian_estimate,
)
from .utils import check_input_type as check_input_type

def minimize_lbfgs(
    objective_func: Any,
    initial_position: Any,
    history_size: int = ...,
    max_iters: int = ...,
    tolerance_grad: float = ...,
    tolerance_change: float = ...,
    initial_inverse_hessian_estimate: Any | None = ...,
    line_search_fn: str = ...,
    max_line_search_iters: int = ...,
    initial_step_length: float = ...,
    dtype: str = ...,
    name: str | None = ...,
): ...
