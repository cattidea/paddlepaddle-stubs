from __future__ import annotations

from typing import Any, Optional

def auto_cast(
    enable: bool = ...,
    custom_white_list: Any | None = ...,
    custom_black_list: Any | None = ...,
    level: str = ...,
    dtype: str = ...,
): ...
def decorate(
    models: Any,
    optimizers: Any | None = ...,
    level: str = ...,
    master_weight: Any | None = ...,
    save_dtype: Any | None = ...,
): ...
