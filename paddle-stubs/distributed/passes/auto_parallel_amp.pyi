from __future__ import annotations

from typing import Any

from .pass_base import PassBase as PassBase
from .pass_base import register_pass as register_pass

world_process_group: Any

class AMPState:
    def __init__(self, block: Any) -> None: ...
    def cast_forward_program(self, dist_context: Any) -> None: ...
    def cast_backward_program(self, params_grads: Any, dist_context: Any) -> None: ...

class AMPPass(PassBase):
    def __init__(self) -> None: ...
