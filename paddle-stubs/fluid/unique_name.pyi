from __future__ import annotations

from typing import Any, Optional

class UniqueNameGenerator:
    ids: Any = ...
    prefix: Any = ...
    def __init__(self, prefix: Optional[Any] = ...) -> None: ...
    def __call__(self, key: Any): ...

class DygraphParameterNameChecker:
    def __init__(self) -> None: ...
    def __call__(self, name: Any): ...

def generate(key: Any): ...
def switch(new_generator: Optional[Any] = ..., new_para_name_checker: Optional[Any] = ...): ...
def guard(new_generator: Optional[Any] = ...) -> None: ...
