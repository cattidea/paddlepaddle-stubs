from __future__ import annotations

from ...fluid.initializer import ConstantInitializer as ConstantInitializer

class Constant(ConstantInitializer):
    def __init__(self, value: float = ...) -> None: ...
