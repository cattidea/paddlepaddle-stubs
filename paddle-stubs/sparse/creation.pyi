from __future__ import annotations

from typing import Any, Optional

def sparse_coo_tensor(
    indices: Any,
    values: Any,
    shape: Optional[Any] = ...,
    dtype: Optional[Any] = ...,
    place: Optional[Any] = ...,
    stop_gradient: bool = ...,
): ...
def sparse_csr_tensor(
    crows: Any,
    cols: Any,
    values: Any,
    shape: Any,
    dtype: Optional[Any] = ...,
    place: Optional[Any] = ...,
    stop_gradient: bool = ...,
): ...
