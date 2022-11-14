from __future__ import annotations

from typing import Any, Optional

def sparse_coo_tensor(
    indices: Any,
    values: Any,
    shape: Any | None = ...,
    dtype: Any | None = ...,
    place: Any | None = ...,
    stop_gradient: bool = ...,
): ...
def sparse_csr_tensor(
    crows: Any,
    cols: Any,
    values: Any,
    shape: Any,
    dtype: Any | None = ...,
    place: Any | None = ...,
    stop_gradient: bool = ...,
): ...
