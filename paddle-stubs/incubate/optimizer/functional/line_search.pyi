from __future__ import annotations

from typing import Any

def cubic_interpolation_(x1: Any, f1: Any, g1: Any, x2: Any, f2: Any, g2: Any): ...
def strong_wolfe(
    f: Any,
    xk: Any,
    pk: Any,
    max_iters: int = ...,
    tolerance_change: float = ...,
    initial_step_length: float = ...,
    c1: float = ...,
    c2: float = ...,
    alpha_max: int = ...,
    dtype: str = ...,
): ...
