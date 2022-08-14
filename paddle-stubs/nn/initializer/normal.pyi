from __future__ import annotations

from typing import Optional

from ...fluid.initializer import NormalInitializer, TruncatedNormalInitializer

class Normal(NormalInitializer):
    def __init__(self, mean: float = ..., std: float = ..., name: Optional[str] = ...) -> None: ...

class TruncatedNormal(TruncatedNormalInitializer):
    def __init__(self, mean: float = ..., std: float = ..., name: Optional[str] = ...) -> None: ...
