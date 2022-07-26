from __future__ import annotations

from typing import Any

from paddle.distribution import distribution

class Independent(distribution.Distribution):
    def __init__(self, base: Any, reinterpreted_batch_rank: Any) -> None: ...
    @property
    def mean(self): ...
    @property
    def variance(self): ...
    def sample(self, shape: Any = ...): ...
    def log_prob(self, value: Any): ...
    def prob(self, value: Any): ...
    def entropy(self): ...
