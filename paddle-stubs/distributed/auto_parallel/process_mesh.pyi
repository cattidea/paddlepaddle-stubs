from __future__ import annotations

from typing import Any

class ProcessMesh:
    def __init__(self, mesh: Any) -> None: ...
    @property
    def topology(self): ...
    @property
    def processes(self): ...
    @property
    def ndim(self): ...
    def __eq__(self, other: object) -> Any: ...
    def __ne__(self, other: object) -> Any: ...
