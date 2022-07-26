from __future__ import annotations

from typing import Any, Optional

from ...fluid.initializer import MSRAInitializer as MSRAInitializer

class KaimingNormal(MSRAInitializer):
    def __init__(self, fan_in: Optional[Any] = ..., negative_slope: float = ..., nonlinearity: str = ...) -> None: ...

class KaimingUniform(MSRAInitializer):
    def __init__(self, fan_in: Optional[Any] = ..., negative_slope: float = ..., nonlinearity: str = ...) -> None: ...
