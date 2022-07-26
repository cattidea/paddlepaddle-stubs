from __future__ import annotations

from typing import Any, Optional

from ...fluid.initializer import NormalInitializer as NormalInitializer
from ...fluid.initializer import (
    TruncatedNormalInitializer as TruncatedNormalInitializer,
)

class Normal(NormalInitializer):
    def __init__(self, mean: float = ..., std: float = ..., name: Optional[Any] = ...) -> None: ...

class TruncatedNormal(TruncatedNormalInitializer):
    def __init__(self, mean: float = ..., std: float = ..., name: Optional[Any] = ...) -> None: ...
