from __future__ import annotations

from typing import Any, Optional

class LoggerFactory:
    @staticmethod
    def build_logger(name: str | None = ..., level: Any = ...): ...

logger: Any

def layer_to_str(base: Any, *args: Any, **kwargs: Any): ...
