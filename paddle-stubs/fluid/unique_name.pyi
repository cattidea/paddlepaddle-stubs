from __future__ import annotations

from typing import Any, Optional

class UniqueNameGenerator:
    ids: Any = ...
    prefix: Any = ...
    def __init__(self, prefix: Any | None = ...) -> None: ...
    def __call__(self, key: Any): ...

class DygraphParameterNameChecker:
    def __init__(self) -> None: ...
    def __call__(self, name: Any): ...

def generate(key: Any): ...
def switch(new_generator: Any | None = ..., new_para_name_checker: Any | None = ...): ...
def guard(new_generator: Any | None = ...) -> None: ...
