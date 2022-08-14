from __future__ import annotations

from typing import Any, Optional

from paddle.fluid import core as core

def graph_sample_neighbors(
    row: Any,
    colptr: Any,
    input_nodes: Any,
    eids: Optional[Any] = ...,
    perm_buffer: Optional[Any] = ...,
    sample_size: int = ...,
    return_eids: bool = ...,
    flag_perm_buffer: bool = ...,
    name: Optional[str] = ...,
): ...
