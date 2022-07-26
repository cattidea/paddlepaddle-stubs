from __future__ import annotations

from typing import Any

class Constraint:
    def __call__(self, value: Any) -> None: ...

class Real(Constraint):
    def __call__(self, value: Any): ...

class Range(Constraint):
    def __init__(self, lower: Any, upper: Any) -> None: ...
    def __call__(self, value: Any): ...

class Positive(Constraint):
    def __call__(self, value: Any): ...

class Simplex(Constraint):
    def __call__(self, value: Any): ...

real: Any
positive: Any
simplex: Any
