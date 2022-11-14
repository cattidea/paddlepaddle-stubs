from __future__ import annotations

from enum import Enum as Enum
from typing import Any, Optional

logger_: Any

def group_sharded_parallel(
    model: Any,
    optimizer: Any,
    level: Any,
    scaler: Any | None = ...,
    group: Any | None = ...,
    offload: bool = ...,
    sync_buffers: bool = ...,
    buffer_max_size: Any = ...,
    segment_size: Any = ...,
    sync_comm: bool = ...,
): ...
def save_group_sharded_model(model: Any, output: Any, optimizer: Any | None = ...) -> None: ...
