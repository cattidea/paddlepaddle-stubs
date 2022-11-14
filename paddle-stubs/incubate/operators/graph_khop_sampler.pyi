from __future__ import annotations

from typing import Any, Optional

from paddle.fluid import core as core

def graph_khop_sampler(
    row: Any,
    colptr: Any,
    input_nodes: Any,
    sample_sizes: Any,
    sorted_eids: Any | None = ...,
    return_eids: bool = ...,
    name: str | None = ...,
): ...
