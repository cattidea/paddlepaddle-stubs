from __future__ import annotations

from typing import Any, Callable, TypeVar

_CallableFunc = TypeVar("_CallableFunc", bound=Callable[..., Any])

def deprecated(
    update_to: str = ..., since: str = ..., reason: str = ..., level: int = ...
) -> Callable[[_CallableFunc], _CallableFunc]: ...
