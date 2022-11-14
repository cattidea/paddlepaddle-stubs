from __future__ import annotations

from collections import namedtuple as namedtuple
from typing import Any, Optional

class ProgressBar:
    file: Any = ...
    name: Any = ...
    def __init__(
        self,
        num: Any | None = ...,
        width: int = ...,
        verbose: int = ...,
        start: bool = ...,
        file: Any = ...,
        name: str = ...,
    ) -> None: ...
    def start(self) -> None: ...
    def update(self, current_num: Any, values: Any = ...): ...
