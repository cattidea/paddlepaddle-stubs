from __future__ import annotations

from typing import Any, Optional

from . import core

class Generator(core.Generator):
    place: Any = ...
    def __init__(self, place: Optional[Any] = ...) -> None: ...
