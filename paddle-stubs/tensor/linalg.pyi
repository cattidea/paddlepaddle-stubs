from __future__ import annotations

from typing import Any, Optional

import numpy as np
from paddle.common_ops_import import core as core
from typing_extensions import Literal

from .. import Tensor
from ..fluid import layers as layers
from ..fluid.data_feeder import check_dtype as check_dtype
from ..fluid.data_feeder import check_type as check_type
from ..fluid.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ..fluid.framework import in_dygraph_mode as in_dygraph_mode
from ..fluid.layer_helper import LayerHelper as LayerHelper
from ..fluid.layers import cast as cast
from ..fluid.layers import transpose as transpose
from ..static import Variable as Variable

K_DEFAULT_DIM: int

def matmul(x: Any, y: Any, transpose_x: bool = ..., transpose_y: bool = ..., name: Optional[Any] = ...): ...
def norm(
    x: Tensor,
    # TODO: Only support "fro", inf, -inf, 0, 1, 2
    p: Literal["fro"] | float = ...,
    axis: Optional[int | list[int] | tuple[int, ...]] = ...,
    keepdim: bool = ...,
    name: Optional[str] = ...,
) -> Tensor: ...
def dist(x: Any, y: Any, p: int = ..., name: Optional[Any] = ...): ...
def cond(
    x: Tensor,
    # TODO: Only support "fro", "nuc", 1, -1, 2, -2, inf, -inf
    #       but now Literal does not support inf
    p: Optional[Literal["fro", "nuc"] | float] = ...,
    name: Optional[str] = ...,
) -> Tensor: ...
def dot(x: Any, y: Any, name: Optional[Any] = ...): ...
def cov(
    x: Tensor,
    rowvar: bool = ...,
    ddof: bool = ...,
    fweights: Optional[Tensor] = ...,
    aweights: Optional[Tensor] = ...,
    name: Optional[str] = ...,
) -> Tensor: ...
def t(input: Any, name: Optional[Any] = ...): ...
def cross(x: Any, y: Any, axis: int = ..., name: Optional[Any] = ...): ...
def cholesky(x: Tensor, upper: bool = ..., name: Optional[str] = ...) -> Tensor: ...
def matrix_rank(
    x: Tensor,
    tol: Optional[float | Tensor] = ...,
    hermitian: bool = ...,
    name: Optional[str] = ...,
) -> Tensor: ...
def bmm(x: Any, y: Any, name: Optional[Any] = ...): ...
def histogram(input: Any, bins: int = ..., min: int = ..., max: int = ..., name: Optional[Any] = ...): ...
def bincount(x: Any, weights: Optional[Any] = ..., minlength: int = ..., name: Optional[Any] = ...): ...
def mv(x: Any, vec: Any, name: Optional[Any] = ...): ...
def det(x: Tensor, name: Optional[str] = ...) -> Tensor: ...
def slogdet(x: Tensor, name: Optional[str] = ...) -> Tensor: ...
def svd(
    x: Tensor,
    full_matrices: bool = ...,
    name: Optional[str] = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...
def matrix_power(x: Tensor, n: int, name: Optional[str] = ...) -> Tensor: ...
def qr(
    x: Tensor,
    mode: Literal["reduced", "complete", "r"] = ...,
    name: Optional[str] = ...,
) -> tuple[Tensor, Tensor]: ...
def lu(
    x: Tensor,
    pivot: bool = ...,
    get_infos: bool = ...,
    name: Optional[str] = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...
def lu_unpack(
    x: Tensor,
    y: Tensor,
    unpack_ludata: bool = ...,
    unpack_pivots: bool = ...,
    name: Optional[str] = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...
def eig(x: Tensor, name: Optional[str] = ...) -> tuple[Tensor, Tensor]: ...
def eigvals(x: Tensor, name: Optional[str] = ...) -> Tensor: ...
def multi_dot(x: list[Tensor], name: Optional[str] = ...) -> Tensor: ...
def eigh(x: Tensor, UPLO: str = ..., name: Optional[str] = ...) -> tuple[Tensor, Tensor]: ...
def pinv(
    x: Tensor,
    rcond: float = ...,
    hermitian: bool = ...,
    name: Optional[str] = ...,
) -> Tensor: ...
def solve(x: Tensor, y: Tensor, name: Optional[str] = ...) -> Tensor: ...
def triangular_solve(
    x: Tensor,
    y: Tensor,
    upper: bool = ...,
    transpose: bool = ...,
    unitriangular: bool = ...,
    name: Optional[str] = ...,
) -> Tensor: ...
def cholesky_solve(x: Tensor, y: Tensor, upper: bool = ..., name: Optional[str] = ...) -> Tensor: ...
def eigvalsh(x: Tensor, UPLO: str = ..., name: Optional[str] = ...) -> Tensor: ...
def lstsq(
    x: Tensor,
    y: Tensor,
    rcond: Optional[float] = ...,
    driver: Optional[Literal["gels", "gelsy", "gelsd", "gelss"]] = ...,
    name: Optional[str] = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
