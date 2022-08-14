# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from paddle import Tensor
from typing_extensions import assert_type


def test_import():
    paddle.nn.AvgPool1D
    paddle.nn.AvgPool2D
    paddle.nn.AvgPool3D
    paddle.nn.MaxPool1D
    paddle.nn.MaxPool2D
    paddle.nn.MaxPool3D
    paddle.nn.AdaptiveAvgPool1D
    paddle.nn.AdaptiveAvgPool2D
    paddle.nn.AdaptiveAvgPool3D
    paddle.nn.AdaptiveMaxPool1D
    paddle.nn.AdaptiveMaxPool2D
    paddle.nn.AdaptiveMaxPool3D
    paddle.nn.MaxUnPool1D
    paddle.nn.MaxUnPool2D
    paddle.nn.MaxUnPool3D

    from paddle.nn import AdaptiveAvgPool1D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import AdaptiveAvgPool2D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import AdaptiveAvgPool3D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import AdaptiveMaxPool1D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import AdaptiveMaxPool2D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import AdaptiveMaxPool3D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import AvgPool1D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import AvgPool2D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import AvgPool3D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import MaxPool1D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import MaxPool2D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import MaxPool3D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import MaxUnPool1D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import MaxUnPool2D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import MaxUnPool3D  # pyright: ignore [reportUnusedImport]


def test_types():
    tensor = paddle.randint(0, 255, shape=[3, 224, 224])

    layer = paddle.nn.AvgPool1D(2)
    assert_type(layer, paddle.nn.AvgPool1D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.AvgPool2D(2, data_format="NCHW")
    assert_type(layer, paddle.nn.AvgPool2D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.AvgPool3D(2, data_format="NCDHW")
    assert_type(layer, paddle.nn.AvgPool3D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.MaxPool1D(2)
    assert_type(layer, paddle.nn.MaxPool1D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.MaxPool2D(2, data_format="NCHW")
    assert_type(layer, paddle.nn.MaxPool2D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.MaxPool3D(2, data_format="NCDHW")
    assert_type(layer, paddle.nn.MaxPool3D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.AdaptiveAvgPool1D(2)
    assert_type(layer, paddle.nn.AdaptiveAvgPool1D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.AdaptiveAvgPool2D(2, data_format="NCHW")
    assert_type(layer, paddle.nn.AdaptiveAvgPool2D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.AdaptiveAvgPool3D(2, data_format="NCDHW")
    assert_type(layer, paddle.nn.AdaptiveAvgPool3D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.AdaptiveMaxPool1D(2)
    assert_type(layer, paddle.nn.AdaptiveMaxPool1D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.AdaptiveMaxPool2D(2)
    assert_type(layer, paddle.nn.AdaptiveMaxPool2D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.AdaptiveMaxPool3D(2)
    assert_type(layer, paddle.nn.AdaptiveMaxPool3D)
    assert_type(layer(tensor), Tensor)

    indices = paddle.to_tensor([])
    layer = paddle.nn.MaxUnPool1D(2)
    assert_type(layer, paddle.nn.MaxUnPool1D)
    assert_type(layer(tensor, indices), Tensor)

    layer = paddle.nn.MaxUnPool2D(2)
    assert_type(layer, paddle.nn.MaxUnPool2D)
    assert_type(layer(tensor, indices), Tensor)

    layer = paddle.nn.MaxUnPool3D(2)
    assert_type(layer, paddle.nn.MaxUnPool3D)
    assert_type(layer(tensor, indices), Tensor)
