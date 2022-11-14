from __future__ import annotations

from typing import Any, Optional

from paddle.fluid import core as core

def graph_reindex(
    x: Any,
    neighbors: Any,
    count: Any,
    value_buffer: Any | None = ...,
    index_buffer: Any | None = ...,
    flag_buffer_hashtable: bool = ...,
    name: str | None = ...,
): ...
