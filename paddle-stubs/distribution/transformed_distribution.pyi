from __future__ import annotations

from typing import Any

from paddle.distribution import distribution

class TransformedDistribution(distribution.Distribution):
    def __init__(self, base: Any, transforms: Any) -> None: ...
    def sample(self, shape: Any = ...): ...
    def log_prob(self, value: Any): ...
