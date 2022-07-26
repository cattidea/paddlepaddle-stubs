from __future__ import annotations

from typing import Any

from . import core as core

MP_STATUS_CHECK_INTERVAL: float
multiprocess_queue_set: Any

class CleanupFuncRegistrar:
    @classmethod
    def register(cls, function: Any, signals: Any = ...) -> None: ...
