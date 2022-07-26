from __future__ import annotations

from typing import Any

class WeightedAverage:
    def __init__(self) -> None: ...
    numerator: Any = ...
    denominator: Any = ...
    def reset(self) -> None: ...
    def add(self, value: Any, weight: Any) -> None: ...
    def eval(self): ...
