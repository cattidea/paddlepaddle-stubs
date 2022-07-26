from __future__ import annotations

from typing import Any, Optional

from paddle.distribution import distribution
from paddle.fluid import core as core
from paddle.fluid.data_feeder import check_dtype as check_dtype
from paddle.fluid.data_feeder import (
    check_variable_and_dtype as check_variable_and_dtype,
)
from paddle.fluid.framework import in_dygraph_mode as in_dygraph_mode
from paddle.fluid.layers import control_flow as control_flow
from paddle.fluid.layers import elementwise_add as elementwise_add
from paddle.fluid.layers import elementwise_div as elementwise_div
from paddle.fluid.layers import elementwise_mul as elementwise_mul
from paddle.fluid.layers import elementwise_sub as elementwise_sub
from paddle.fluid.layers import nn as nn
from paddle.tensor import arange as arange
from paddle.tensor import concat as concat
from paddle.tensor import gather_nd as gather_nd

class Categorical(distribution.Distribution):
    name: Any = ...
    dtype: str = ...
    logits: Any = ...
    def __init__(self, logits: Any, name: Optional[Any] = ...) -> None: ...
    def sample(self, shape: Any): ...
    def kl_divergence(self, other: Any): ...
    def entropy(self): ...
    def probs(self, value: Any): ...
    def log_prob(self, value: Any): ...
