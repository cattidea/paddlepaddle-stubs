from __future__ import annotations

from typing import Any, Optional

from paddle.base import core as core

def graph_sample_neighbors(
    row: Any,
    colptr: Any,
    input_nodes: Any,
    eids: Any | None = ...,
    perm_buffer: Any | None = ...,
    sample_size: int = ...,
    return_eids: bool = ...,
    flag_perm_buffer: bool = ...,
    name: str | None = ...,
): ...
