from __future__ import annotations

from typing import Any, Optional

from ...fluid.initializer import UniformInitializer as UniformInitializer

class Uniform(UniformInitializer):
    def __init__(self, low: Any = ..., high: float = ..., name: Optional[Any] = ...) -> None: ...
