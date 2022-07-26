from __future__ import annotations

from ...ps.the_one_ps import TheOnePSRuntime as TheOnePSRuntime
from ..runtime.collective_runtime import CollectiveRuntime as CollectiveRuntime
from ..runtime.parameter_server_runtime import (
    ParameterServerRuntime as ParameterServerRuntime,
)

class RuntimeFactory:
    def __init__(self) -> None: ...
