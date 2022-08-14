# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from paddle import Tensor
from typing_extensions import assert_type


def test_import():
    paddle.nn.PixelShuffle

    from paddle.nn import PixelShuffle  # pyright: ignore [reportUnusedImport]


def test_types():
    tensor = paddle.randint(0, 255, shape=[3, 224, 224])

    layer = paddle.nn.PixelShuffle(3)
    assert_type(layer, paddle.nn.PixelShuffle)
    assert_type(layer(tensor), Tensor)
