from __future__ import annotations

from typing import Any, Optional

from google.protobuf import text_format as text_format

from ..backward import append_backward as append_backward
from .node import DownpourServer as DownpourServer
from .node import DownpourWorker as DownpourWorker

class DownpourSGD:
    learning_rate_: Any = ...
    window_: Any = ...
    type: str = ...
    data_norm_name: Any = ...
    def __init__(self, learning_rate: float = ..., window: int = ...) -> None: ...
    def minimize(
        self,
        losses: Any,
        startup_program: Optional[Any] = ...,
        parameter_list: Optional[Any] = ...,
        no_grad_set: Optional[Any] = ...,
    ): ...
