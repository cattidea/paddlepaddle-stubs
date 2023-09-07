from __future__ import annotations

from typing import Any

from paddle.base.data_feeder import check_dtype as check_dtype

def check_input_type(input: Any, name: Any, op_name: Any) -> None: ...
def check_initial_inverse_hessian_estimate(H0: Any): ...
