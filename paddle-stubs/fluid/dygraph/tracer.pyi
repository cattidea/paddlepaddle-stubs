from __future__ import annotations

from collections import defaultdict as defaultdict
from typing import Any, Optional

from paddle.fluid import core

final_state_name_mapping: Any

class Tracer(core.Tracer):
    def __init__(self) -> None: ...
    def eager_trace_op(
        self,
        type: Any,
        inputs: Any,
        outputs: Any,
        attrs: Any,
        stop_gradient: bool = ...,
        inplace_map: Optional[Any] = ...,
    ) -> None: ...
    def eager_final_state_trace_op(
        self,
        type: Any,
        inputs: Any,
        outputs: Any,
        attrs: Any,
        stop_gradient: bool = ...,
        inplace_map: Optional[Any] = ...,
    ) -> None: ...
    def trace_op(
        self,
        type: Any,
        inputs: Any,
        outputs: Any,
        attrs: Any,
        stop_gradient: bool = ...,
        inplace_map: Optional[Any] = ...,
    ) -> None: ...
    def train_mode(self) -> None: ...
    def eval_mode(self) -> None: ...
