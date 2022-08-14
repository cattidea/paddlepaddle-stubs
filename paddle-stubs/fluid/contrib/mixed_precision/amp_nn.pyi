from __future__ import annotations

from typing import Any, Optional

def check_finite_and_unscale(x: Any, scale: Any, name: Optional[str] = ..., float_status: Optional[Any] = ...): ...
def update_loss_scaling(
    x: Any,
    found_inf: Any,
    prev_loss_scaling: Any,
    num_good_steps: Any,
    num_bad_steps: Any,
    incr_every_n_steps: Any,
    decr_every_n_nan_or_inf: Any,
    incr_ratio: Any,
    decr_ratio: Any,
    stop_update: bool = ...,
    name: Optional[str] = ...,
): ...
