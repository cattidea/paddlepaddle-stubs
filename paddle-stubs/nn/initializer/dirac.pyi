from __future__ import annotations

from typing import Any, Optional

from ...fluid.initializer import Initializer

class Dirac(Initializer):
    def __init__(self, groups: int = ..., name: Optional[Any] = ...) -> None: ...
    def __call__(self, var: Any, block: Optional[Any] = ...): ...
