from __future__ import annotations

from typing import Any, Optional

def auto_cast(
    enable: bool = ...,
    custom_white_list: Optional[Any] = ...,
    custom_black_list: Optional[Any] = ...,
    level: str = ...,
    dtype: str = ...,
): ...
def decorate(
    models: Any,
    optimizers: Optional[Any] = ...,
    level: str = ...,
    master_weight: Optional[Any] = ...,
    save_dtype: Optional[Any] = ...,
): ...
