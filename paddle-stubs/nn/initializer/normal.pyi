from __future__ import annotations

from typing import Optional

from ...base.initializer import NormalInitializer, TruncatedNormalInitializer

class Normal(NormalInitializer):
    def __init__(self, mean: float = ..., std: float = ..., name: str | None = ...) -> None: ...

class TruncatedNormal(TruncatedNormalInitializer):
    def __init__(self, mean: float = ..., std: float = ..., name: str | None = ...) -> None: ...
