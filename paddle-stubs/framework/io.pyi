from __future__ import annotations

from typing import Any

def save(obj: Any, path: Any, protocol: int = ..., **configs: Any) -> None: ...
def load(path: Any, **configs: Any): ...
