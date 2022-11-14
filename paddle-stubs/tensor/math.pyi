from __future__ import annotations

from typing import Any, Optional

from .. import Tensor
from ..fluid.data_feeder import check_dtype as check_dtype
from ..fluid.data_feeder import check_type as check_type
from ..fluid.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ..fluid.data_feeder import convert_dtype as convert_dtype
from ..fluid.dygraph.inplace_utils import (
    inplace_apis_in_dygraph_only as inplace_apis_in_dygraph_only,
)
from ..fluid.framework import in_dygraph_mode as in_dygraph_mode
from ..fluid.layer_helper import LayerHelper as LayerHelper

def abs(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def acos(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def acosh(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def asin(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def asinh(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def atan(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def atanh(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def ceil(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def ceil_(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def cos(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def cosh(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def erf(x: Tensor, name: str | None = ...) -> Tensor: ...
def exp(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def exp_(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def expm1(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def floor(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def floor_(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def lgamma(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def log(x: Tensor, name: str | None = ...) -> Tensor: ...
def multiplex(inputs: list[Tensor], index: Tensor, name: str | None = ...) -> Tensor: ...
def reciprocal(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def reciprocal_(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def round(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def round_(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def rsqrt(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def rsqrt_(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def scale(
    x: Tensor,
    scale: float | Tensor = ...,
    bias: float = ...,
    bias_after_scale: bool = ...,
    act: str | None = ...,
    name: str | None = ...,
) -> Tensor: ...
def sin(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def sinh(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def sqrt(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def sqrt_(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def square(x: Tensor, with_quant_attr: bool, name: str | None = ...) -> Tensor: ...
def stanh(x: Tensor, scale_a: float = ..., scale_b: float = ..., name=str | None) -> Tensor: ...
def tan(x: Tensor, scale_a: float = ..., scale_b: float = ..., name=str | None) -> Tensor: ...

from ..fluid.layers.layer_function_generator import (
    generate_activation_fn as generate_activation_fn,
)
from ..fluid.layers.layer_function_generator import (
    generate_layer_fn as generate_layer_fn,
)
from ..framework import convert_np_dtype_to_dtype_ as convert_np_dtype_to_dtype_
from ..framework import core as core

def scale_(
    x: Any,
    scale: float = ...,
    bias: float = ...,
    bias_after_scale: bool = ...,
    act: Any | None = ...,
    name: str | None = ...,
): ...
def pow(x: Any, y: Any, name: str | None = ...): ...

OP_NAMEMAPPING: Any

def add(x: Any, y: Any, name: str | None = ...): ...
def add_(x: Any, y: Any, name: str | None = ...): ...
def subtract(x: Any, y: Any, name: str | None = ...): ...
def subtract_(x: Any, y: Any, name: str | None = ...): ...
def divide(x: Any, y: Any, name: str | None = ...): ...
def floor_divide(x: Any, y: Any, name: str | None = ...): ...
def remainder(x: Any, y: Any, name: str | None = ...): ...

mod = remainder
floor_mod = remainder

def multiply(x: Any, y: Any, name: str | None = ...): ...
def maximum(x: Any, y: Any, name: str | None = ...): ...
def minimum(x: Any, y: Any, name: str | None = ...): ...
def fmax(x: Any, y: Any, name: str | None = ...): ...
def fmin(x: Any, y: Any, name: str | None = ...): ...

proto_dict: Any
op_proto: Any
additional_args_lines: Any

def sum(x: Any, axis: Any | None = ..., dtype: Any | None = ..., keepdim: bool = ..., name: str | None = ...): ...
def nansum(x: Any, axis: Any | None = ..., dtype: Any | None = ..., keepdim: bool = ..., name: str | None = ...): ...
def nanmean(x: Any, axis: Any | None = ..., keepdim: bool = ..., name: str | None = ...): ...
def add_n(inputs: Any, name: str | None = ...): ...
def trunc(input: Any, name: str | None = ...): ...
def mm(input: Any, mat2: Any, name: str | None = ...): ...
def addmm(input: Any, x: Any, y: Any, beta: float = ..., alpha: float = ..., name: str | None = ...): ...
def renorm(x: Any, p: Any, axis: Any, max_norm: Any): ...
def inner(x: Any, y: Any, name: str | None = ...): ...
def outer(x: Any, y: Any, name: str | None = ...): ...
def logsumexp(x: Any, axis: Any | None = ..., keepdim: bool = ..., name: str | None = ...): ...
def inverse(x: Tensor, name: str | None = ...) -> Tensor: ...
def max(x: Any, axis: Any | None = ..., keepdim: bool = ..., name: str | None = ...): ...
def min(x: Any, axis: Any | None = ..., keepdim: bool = ..., name: str | None = ...): ...
def amax(x: Any, axis: Any | None = ..., keepdim: bool = ..., name: str | None = ...): ...
def amin(x: Any, axis: Any | None = ..., keepdim: bool = ..., name: str | None = ...): ...
def log1p(x: Any, name: str | None = ...): ...
def log2(x: Any, name: str | None = ...): ...
def log10(x: Any, name: str | None = ...): ...
def clip(x: Any, min: Any | None = ..., max: Any | None = ..., name: str | None = ...): ...
def clip_(x: Any, min: Any | None = ..., max: Any | None = ..., name: str | None = ...): ...
def trace(x: Any, offset: int = ..., axis1: int = ..., axis2: int = ..., name: str | None = ...): ...
def diagonal(x: Any, offset: int = ..., axis1: int = ..., axis2: int = ..., name: str | None = ...): ...
def kron(x: Any, y: Any, name: str | None = ...): ...
def cumsum(x: Any, axis: Any | None = ..., dtype: Any | None = ..., name: str | None = ...): ...
def cumprod(x: Any, dim: Any | None = ..., dtype: Any | None = ..., name: str | None = ...): ...
def isfinite(x: Any, name: str | None = ...): ...
def isinf(x: Any, name: str | None = ...): ...
def isnan(x: Any, name: str | None = ...): ...
def prod(x: Any, axis: Any | None = ..., keepdim: bool = ..., dtype: Any | None = ..., name: str | None = ...): ...
def sign(x: Any, name: str | None = ...): ...
def tanh(x: Tensor, name: str | None = ...) -> Tensor: ...
def tanh_(x: Tensor, name: str | None = ...) -> Tensor: ...
def increment(x: Any, value: float = ..., name: str | None = ...): ...
def all(x: Any, axis: Any | None = ..., keepdim: bool = ..., name: str | None = ...): ...
def any(x: Any, axis: Any | None = ..., keepdim: bool = ..., name: str | None = ...): ...
def broadcast_shape(x_shape: Any, y_shape: Any): ...
def conj(x: Any, name: str | None = ...): ...
def digamma(x: Any, name: str | None = ...): ...
def neg(x: Any, name: str | None = ...): ...
def atan2(x: Any, y: Any, name: str | None = ...): ...
def logit(x: Any, eps: Any | None = ..., name: str | None = ...): ...
def lerp(x: Any, y: Any, weight: Any, name: str | None = ...): ...
def lerp_(x: Any, y: Any, weight: Any, name: str | None = ...): ...
def erfinv(x: Any, name: str | None = ...): ...
def erfinv_(x: Any, name: str | None = ...): ...
def rad2deg(x: Any, name: str | None = ...): ...
def deg2rad(x: Any, name: str | None = ...): ...
def gcd(x: Any, y: Any, name: str | None = ...): ...
def lcm(x: Any, y: Any, name: str | None = ...): ...
def diff(
    x: Any,
    n: int = ...,
    axis: int = ...,
    prepend: Any | None = ...,
    append: Any | None = ...,
    name: str | None = ...,
): ...
def angle(x: Any, name: str | None = ...): ...
