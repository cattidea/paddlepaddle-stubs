from __future__ import annotations

from typing import Any, Optional

def one_hot(input: Any, depth: Any, allow_out_of_range: bool = ...): ...
def embedding(
    input: Any,
    size: Any,
    is_sparse: bool = ...,
    is_distributed: bool = ...,
    padding_idx: Optional[Any] = ...,
    param_attr: Optional[Any] = ...,
    dtype: str = ...,
): ...
