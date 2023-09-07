from __future__ import annotations

from typing import Any

class Distribution:
    def sample(self) -> None: ...
    def entropy(self) -> None: ...
    def kl_divergence(self, other: Any) -> None: ...
    def log_prob(self, value: Any) -> None: ...

class Uniform(Distribution):
    all_arg_is_float: bool = ...
    batch_size_unknown: bool = ...
    low: Any = ...
    high: Any = ...
    def __init__(self, low: Any, high: Any) -> None: ...
    def sample(self, shape: Any, seed: int = ...): ...
    def log_prob(self, value: Any): ...
    def entropy(self): ...

class Normal(Distribution):
    batch_size_unknown: bool = ...
    all_arg_is_float: bool = ...
    loc: Any = ...
    scale: Any = ...
    def __init__(self, loc: Any, scale: Any) -> None: ...
    def sample(self, shape: Any, seed: int = ...): ...
    def entropy(self): ...
    def log_prob(self, value: Any): ...
    def kl_divergence(self, other: Any): ...

class Categorical(Distribution):
    logits: Any = ...
    def __init__(self, logits: Any) -> None: ...
    def kl_divergence(self, other: Any): ...
    def entropy(self): ...

class MultivariateNormalDiag(Distribution):
    loc: Any = ...
    scale: Any = ...
    def __init__(self, loc: Any, scale: Any) -> None: ...
    def entropy(self): ...
    def kl_divergence(self, other: Any): ...