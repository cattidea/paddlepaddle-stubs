from __future__ import annotations

from paddle.base.framework import Parameter as Parameter

from ..base.private_helper_function import wait_server_ready as wait_server_ready
from .runtime_base import RuntimeBase as RuntimeBase

class ParameterServerRuntime(RuntimeBase):
    def __init__(self) -> None: ...
    def build_compiled_startegy(self): ...
