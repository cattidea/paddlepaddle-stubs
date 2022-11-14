from __future__ import annotations

from typing import Optional

from ...fluid.initializer import XavierInitializer

class XavierNormal(XavierInitializer):
    def __init__(
        self, fan_in: float | None = ..., fan_out: float | None = ..., name: str | None = ...
    ) -> None: ...

class XavierUniform(XavierInitializer):
    def __init__(
        self, fan_in: float | None = ..., fan_out: float | None = ..., name: str | None = ...
    ) -> None: ...
