from __future__ import annotations

from typing import Any, Optional

from ..base.private_helper_function import wait_server_ready as wait_server_ready

OpRole: Any
OP_ROLE_KEY: Any
OP_ROLE_VAR_KEY: Any

def is_update_op(op: Any): ...
def is_loss_grad_op(op: Any): ...
def is_backward_op(op: Any): ...
def is_optimizer_op(op: Any): ...

class CollectiveHelper:
    nrings: Any = ...
    wait_port: Any = ...
    role_maker: Any = ...
    def __init__(self, role_maker: Any, nrings: int = ..., wait_port: bool = ...) -> None: ...
    startup_program: Any = ...
    def update_startup_program(self, startup_program: Any | None = ...) -> None: ...
