from __future__ import annotations

from typing import Any, Optional

from paddle.base import core as core
from paddle.base.data_feeder import check_dtype as check_dtype
from paddle.base.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from paddle.base.framework import in_dygraph_mode as in_dygraph_mode
from paddle.base.layers import control_flow as control_flow
from paddle.base.layers import elementwise_mul as elementwise_mul
from paddle.distribution import distribution

class Normal(distribution.Distribution):
    batch_size_unknown: bool = ...
    all_arg_is_float: bool = ...
    name: Any = ...
    dtype: str = ...
    loc: Any = ...
    scale: Any = ...
    def __init__(self, loc: Any, scale: Any, name: str | None = ...) -> None: ...
    def sample(self, shape: Any, seed: int = ...): ...
    def entropy(self): ...
    def log_prob(self, value: Any): ...
    def probs(self, value: Any): ...
    def kl_divergence(self, other: Any): ...
