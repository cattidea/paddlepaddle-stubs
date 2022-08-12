# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from paddle import Tensor
from typing_extensions import assert_type


def test_import():
    paddle.nn.PairwiseDistance

    from paddle.nn import PairwiseDistance  # pyright: ignore [reportUnusedImport]


def test_types():
    x = paddle.randint(0, 255, shape=[3, 224, 224])
    y = paddle.randint(0, 255, shape=[3, 224, 224])

    layer = paddle.nn.PairwiseDistance(2, 1e-6)
    assert_type(layer, paddle.nn.PairwiseDistance)
    assert_type(layer(x, y), Tensor)
