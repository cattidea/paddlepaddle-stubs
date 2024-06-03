# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing_extensions import assert_type

import paddle
from paddle import Tensor


def test_import():
    paddle.einsum

    from paddle import einsum  # pyright: ignore [reportUnusedImport]


def test_types():
    x = paddle.randn([8, 8], dtype=paddle.float64)
    y = paddle.einsum("ij,jk->ik", x, x)
    assert_type(y, Tensor)
