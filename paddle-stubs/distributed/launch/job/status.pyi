from __future__ import annotations

class Status:
    UNINIT: str = ...
    READY: str = ...
    RUNNING: str = ...
    FAILED: str = ...
    TERMINATING: str = ...
    RESTARTING: str = ...
    UNKNOWN: str = ...
    COMPLETED: str = ...
