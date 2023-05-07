from __future__ import annotations

from typing import Any, Callable, TypeVar

CallableFunc = TypeVar("CallableFunc", bound=Callable[..., Any])

def deprecated(
    update_to: str = ..., since: str = ..., reason: str = ..., level: int = ...
) -> Callable[[CallableFunc], CallableFunc]: ...
