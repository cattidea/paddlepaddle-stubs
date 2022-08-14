from __future__ import annotations

from typing import Optional

from ...fluid.initializer import UniformInitializer

class Uniform(UniformInitializer):
    def __init__(self, low: float = ..., high: float = ..., name: Optional[str] = ...) -> None: ...
