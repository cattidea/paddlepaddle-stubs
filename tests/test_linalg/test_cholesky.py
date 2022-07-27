# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing import Any

import numpy as np
import paddle
from paddle import Tensor
from typing_extensions import assert_type


def test_full_path_access():
    paddle.linalg.cholesky


def test_full_path_import():
    from paddle.linalg import cholesky  # pyright: ignore [reportUnusedImport]


def test_types():
    a = np.random.rand(3, 3)
    a_t: np.ndarray[Any, np.dtype[np.float64]] = np.transpose(a, [1, 0])  # type: ignore
    x_data: np.ndarray[Any, np.dtype[np.float64]] = np.matmul(a, a_t) + 1e-03
    x = paddle.to_tensor(x_data)
    out = paddle.linalg.cholesky(x, upper=False)
    assert_type(out, Tensor)
