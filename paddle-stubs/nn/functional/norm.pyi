from __future__ import annotations

from typing import Any, Optional

from paddle.base.framework import core as core

from ...base import dygraph_utils as dygraph_utils
from ...base.data_feeder import check_type as check_type
from ...base.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ...base.layer_helper import LayerHelper as LayerHelper
from ...framework import ParamAttr as ParamAttr
from ...framework import create_parameter as create_parameter
from ..initializer import Constant as Constant

def normalize(x: Any, p: int = ..., axis: int = ..., epsilon: float = ..., name: str | None = ...): ...
def batch_norm(
    x: Any,
    running_mean: Any,
    running_var: Any,
    weight: Any,
    bias: Any,
    training: bool = ...,
    momentum: float = ...,
    epsilon: float = ...,
    data_format: str = ...,
    use_global_stats: Any | None = ...,
    name: str | None = ...,
): ...
def layer_norm(
    x: Any,
    normalized_shape: Any,
    weight: Any | None = ...,
    bias: Any | None = ...,
    epsilon: float = ...,
    name: str | None = ...,
): ...
def instance_norm(
    x: Any,
    running_mean: Any | None = ...,
    running_var: Any | None = ...,
    weight: Any | None = ...,
    bias: Any | None = ...,
    use_input_stats: bool = ...,
    momentum: float = ...,
    eps: float = ...,
    data_format: str = ...,
    name: str | None = ...,
): ...
def local_response_norm(
    x: Any,
    size: Any,
    alpha: float = ...,
    beta: float = ...,
    k: float = ...,
    data_format: str = ...,
    name: str | None = ...,
): ...
