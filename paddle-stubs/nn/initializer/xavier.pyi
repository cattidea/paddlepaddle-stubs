from __future__ import annotations

from typing import Optional

from ...fluid.initializer import XavierInitializer

class XavierNormal(XavierInitializer):
    def __init__(
        self, fan_in: Optional[float] = ..., fan_out: Optional[float] = ..., name: Optional[str] = ...
    ) -> None: ...

class XavierUniform(XavierInitializer):
    def __init__(
        self, fan_in: Optional[float] = ..., fan_out: Optional[float] = ..., name: Optional[str] = ...
    ) -> None: ...
