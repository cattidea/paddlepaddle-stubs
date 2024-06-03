# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing_extensions import assert_type

import paddle
from paddle import Tensor


def test_import():
    paddle.nn.Conv1D
    paddle.nn.Conv1DTranspose
    paddle.nn.Conv2D
    paddle.nn.Conv2DTranspose
    paddle.nn.Conv3D
    paddle.nn.Conv3DTranspose

    from paddle.nn import (
        Conv1D,  # pyright: ignore [reportUnusedImport]
        Conv1DTranspose,  # pyright: ignore [reportUnusedImport]
        Conv2D,  # pyright: ignore [reportUnusedImport]
        Conv2DTranspose,  # pyright: ignore [reportUnusedImport]
        Conv3D,  # pyright: ignore [reportUnusedImport]
        Conv3DTranspose,  # pyright: ignore [reportUnusedImport]
    )


def test_types():
    tensor = paddle.randint(0, 255, shape=[3, 224, 224])

    layer = paddle.nn.Conv1D(1, 2, 3)
    assert_type(layer, paddle.nn.Conv1D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Conv1DTranspose(1, 2, 3)
    assert_type(layer, paddle.nn.Conv1DTranspose)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Conv2D(1, 2, 3)
    assert_type(layer, paddle.nn.Conv2D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Conv2DTranspose(1, 2, 3)
    assert_type(layer, paddle.nn.Conv2DTranspose)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Conv3D(1, 2, 3)
    assert_type(layer, paddle.nn.Conv3D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Conv3DTranspose(1, 2, 3)
    assert_type(layer, paddle.nn.Conv3DTranspose)
    assert_type(layer(tensor), Tensor)
