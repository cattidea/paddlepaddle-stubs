from __future__ import annotations

from typing import Any, Optional

from ...fluid.initializer import XavierInitializer as XavierInitializer

class XavierNormal(XavierInitializer):
    def __init__(
        self, fan_in: Optional[Any] = ..., fan_out: Optional[Any] = ..., name: Optional[Any] = ...
    ) -> None: ...

class XavierUniform(XavierInitializer):
    def __init__(
        self, fan_in: Optional[Any] = ..., fan_out: Optional[Any] = ..., name: Optional[Any] = ...
    ) -> None: ...
