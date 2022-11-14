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

def matmul(x: Any, y: Any, transpose_x: bool = ..., transpose_y: bool = ..., name: str | None = ...): ...
def norm(
    x: Tensor,
    # TODO: Only support "fro", inf, -inf, 0, 1, 2
    p: Literal["fro"] | float = ...,
    axis: int | list[int] | tuple[int, ...] | None = ...,
    keepdim: bool = ...,
    name: str | None = ...,
) -> Tensor: ...
def dist(x: Any, y: Any, p: int = ..., name: str | None = ...): ...
def cond(
    x: Tensor,
    # TODO: Only support "fro", "nuc", 1, -1, 2, -2, inf, -inf
    #       but now Literal does not support inf
    p: Literal['fro', 'nuc'] | float | None = ...,
    name: str | None = ...,
) -> Tensor: ...
def dot(x: Any, y: Any, name: str | None = ...): ...
def cov(
    x: Tensor,
    rowvar: bool = ...,
    ddof: bool = ...,
    fweights: Tensor | None = ...,
    aweights: Tensor | None = ...,
    name: str | None = ...,
) -> Tensor: ...
def t(input: Any, name: str | None = ...): ...
def cross(x: Any, y: Any, axis: int = ..., name: str | None = ...): ...
def cholesky(x: Tensor, upper: bool = ..., name: str | None = ...) -> Tensor: ...
def matrix_rank(
    x: Tensor,
    tol: float | Tensor | None = ...,
    hermitian: bool = ...,
    name: str | None = ...,
) -> Tensor: ...
def bmm(x: Any, y: Any, name: str | None = ...): ...
def histogram(input: Any, bins: int = ..., min: int = ..., max: int = ..., name: str | None = ...): ...
def bincount(x: Any, weights: Any | None = ..., minlength: int = ..., name: str | None = ...): ...
def mv(x: Any, vec: Any, name: str | None = ...): ...
def det(x: Tensor, name: str | None = ...) -> Tensor: ...
def slogdet(x: Tensor, name: str | None = ...) -> Tensor: ...
def svd(
    x: Tensor,
    full_matrices: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...
def matrix_power(x: Tensor, n: int, name: str | None = ...) -> Tensor: ...
def qr(
    x: Tensor,
    mode: Literal["reduced", "complete", "r"] = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor]: ...
def lu(
    x: Tensor,
    pivot: bool = ...,
    get_infos: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...
def lu_unpack(
    x: Tensor,
    y: Tensor,
    unpack_ludata: bool = ...,
    unpack_pivots: bool = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor, Tensor]: ...
def eig(x: Tensor, name: str | None = ...) -> tuple[Tensor, Tensor]: ...
def eigvals(x: Tensor, name: str | None = ...) -> Tensor: ...
def multi_dot(x: list[Tensor], name: str | None = ...) -> Tensor: ...
def eigh(x: Tensor, UPLO: str = ..., name: str | None = ...) -> tuple[Tensor, Tensor]: ...
def pinv(
    x: Tensor,
    rcond: float = ...,
    hermitian: bool = ...,
    name: str | None = ...,
) -> Tensor: ...
def solve(x: Tensor, y: Tensor, name: str | None = ...) -> Tensor: ...
def triangular_solve(
    x: Tensor,
    y: Tensor,
    upper: bool = ...,
    transpose: bool = ...,
    unitriangular: bool = ...,
    name: str | None = ...,
) -> Tensor: ...
def cholesky_solve(x: Tensor, y: Tensor, upper: bool = ..., name: str | None = ...) -> Tensor: ...
def eigvalsh(x: Tensor, UPLO: str = ..., name: str | None = ...) -> Tensor: ...
def lstsq(
    x: Tensor,
    y: Tensor,
    rcond: float | None = ...,
    driver: Literal['gels', 'gelsy', 'gelsd', 'gelss'] | None = ...,
    name: str | None = ...,
) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...
