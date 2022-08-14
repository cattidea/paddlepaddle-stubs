from __future__ import annotations

from typing import Any, Optional

from ...fluid.framework import default_main_program as default_main_program

def sparse_attention(
    query: Any,
    key: Any,
    value: Any,
    sparse_csr_offset: Any,
    sparse_csr_columns: Any,
    key_padding_mask: Optional[Any] = ...,
    attn_mask: Optional[Any] = ...,
    name: Optional[str] = ...,
): ...
