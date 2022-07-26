from __future__ import annotations

from typing import Any, Optional

def amp_guard(
    enable: bool = ...,
    custom_white_list: Optional[Any] = ...,
    custom_black_list: Optional[Any] = ...,
    level: str = ...,
    dtype: str = ...,
) -> None: ...

class StateDictHook:
    def __init__(self, save_dtype: Any) -> None: ...
    def __call__(self, state_dict: Any) -> None: ...

def amp_decorate(
    models: Any,
    optimizers: Optional[Any] = ...,
    level: str = ...,
    master_weight: Optional[Any] = ...,
    save_dtype: Optional[Any] = ...,
): ...
