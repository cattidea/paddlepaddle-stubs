from __future__ import annotations

from typing import Any, Optional

from paddle.fluid.framework import core as core

from ...fluid import dygraph_utils as dygraph_utils
from ...fluid.data_feeder import check_type as check_type
from ...fluid.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ...fluid.layer_helper import LayerHelper as LayerHelper
from ...framework import ParamAttr as ParamAttr
from ...framework import create_parameter as create_parameter
from ..initializer import Constant as Constant

def normalize(x: Any, p: int = ..., axis: int = ..., epsilon: float = ..., name: Optional[Any] = ...): ...
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
    use_global_stats: Optional[Any] = ...,
    name: Optional[Any] = ...,
): ...
def layer_norm(
    x: Any,
    normalized_shape: Any,
    weight: Optional[Any] = ...,
    bias: Optional[Any] = ...,
    epsilon: float = ...,
    name: Optional[Any] = ...,
): ...
def instance_norm(
    x: Any,
    running_mean: Optional[Any] = ...,
    running_var: Optional[Any] = ...,
    weight: Optional[Any] = ...,
    bias: Optional[Any] = ...,
    use_input_stats: bool = ...,
    momentum: float = ...,
    eps: float = ...,
    data_format: str = ...,
    name: Optional[Any] = ...,
): ...
def local_response_norm(
    x: Any,
    size: Any,
    alpha: float = ...,
    beta: float = ...,
    k: float = ...,
    data_format: str = ...,
    name: Optional[Any] = ...,
): ...
