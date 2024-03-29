from __future__ import annotations

from typing import Any

from paddle.base import core as core
from paddle.base.data_feeder import check_dtype as check_dtype
from paddle.base.data_feeder import check_type as check_type
from paddle.base.framework import in_dygraph_mode as in_dygraph_mode
from paddle.base.layers import control_flow as control_flow
from paddle.base.layers import elementwise_add as elementwise_add
from paddle.base.layers import elementwise_div as elementwise_div
from paddle.base.layers import elementwise_mul as elementwise_mul
from paddle.base.layers import elementwise_sub as elementwise_sub
from paddle.base.layers import nn as nn
from paddle.base.layers import ops as ops
from paddle.tensor import arange as arange
from paddle.tensor import concat as concat
from paddle.tensor import gather_nd as gather_nd
from paddle.tensor import multinomial as multinomial

class Distribution:
    def __init__(self, batch_shape: Any = ..., event_shape: Any = ...) -> None: ...
    @property
    def batch_shape(self): ...
    @property
    def event_shape(self): ...
    @property
    def mean(self) -> None: ...
    @property
    def variance(self) -> None: ...
    def sample(self, shape: Any = ...) -> None: ...
    def rsample(self, shape: Any = ...) -> None: ...
    def entropy(self) -> None: ...
    def kl_divergence(self, other: Any) -> None: ...
    def prob(self, value: Any): ...
    def log_prob(self, value: Any) -> None: ...
    def probs(self, value: Any) -> None: ...
