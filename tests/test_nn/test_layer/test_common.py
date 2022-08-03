# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from paddle import Tensor
from typing_extensions import assert_type


def test_import():
    paddle.nn.Flatten
    paddle.nn.Identity
    paddle.nn.Linear
    paddle.nn.Upsample
    paddle.nn.UpsamplingNearest2D
    paddle.nn.UpsamplingBilinear2D
    paddle.nn.Bilinear
    paddle.nn.Dropout
    paddle.nn.Dropout2D
    paddle.nn.Dropout3D
    paddle.nn.AlphaDropout
    paddle.nn.Pad1D
    paddle.nn.Pad2D
    paddle.nn.ZeroPad2D
    paddle.nn.Pad3D
    paddle.nn.CosineSimilarity
    paddle.nn.Embedding
    paddle.nn.Unfold
    paddle.nn.Fold

    from paddle.nn import AlphaDropout  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Bilinear  # pyright: ignore [reportUnusedImport]
    from paddle.nn import CosineSimilarity  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Dropout  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Dropout2D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Dropout3D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Embedding  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Flatten  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Fold  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Identity  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Linear  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Pad1D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Pad2D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Pad3D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Unfold  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Upsample  # pyright: ignore [reportUnusedImport]
    from paddle.nn import UpsamplingBilinear2D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import UpsamplingNearest2D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import ZeroPad2D  # pyright: ignore [reportUnusedImport]


def test_types():
    tensor = paddle.to_tensor([1, 2, 3])

    layer = paddle.nn.Flatten()
    assert_type(layer, paddle.nn.Flatten)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Identity()
    assert_type(layer, paddle.nn.Identity)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Linear(3, 4)
    assert_type(layer, paddle.nn.Linear)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Upsample(scale_factor=2)
    assert_type(layer, paddle.nn.Upsample)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.UpsamplingNearest2D(scale_factor=2)
    assert_type(layer, paddle.nn.UpsamplingNearest2D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.UpsamplingBilinear2D(scale_factor=2)
    assert_type(layer, paddle.nn.UpsamplingBilinear2D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Bilinear(2, 2, 2)
    assert_type(layer, paddle.nn.Bilinear)
    assert_type(layer(tensor, tensor), Tensor)

    layer = paddle.nn.Dropout(p=0.5)
    assert_type(layer, paddle.nn.Dropout)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Dropout2D(p=0.5)
    assert_type(layer, paddle.nn.Dropout2D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Dropout3D(p=0.5)
    assert_type(layer, paddle.nn.Dropout3D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.AlphaDropout(p=0.5)
    assert_type(layer, paddle.nn.AlphaDropout)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Pad1D(padding=1)
    assert_type(layer, paddle.nn.Pad1D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Pad2D(padding=(1, 1))
    assert_type(layer, paddle.nn.Pad2D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.ZeroPad2D(padding=(1, 1))
    assert_type(layer, paddle.nn.ZeroPad2D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Pad3D(padding=(1, 1, 1))
    assert_type(layer, paddle.nn.Pad3D)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.CosineSimilarity()
    assert_type(layer, paddle.nn.CosineSimilarity)
    assert_type(layer(tensor, tensor), Tensor)

    layer = paddle.nn.Embedding(3, 4)
    assert_type(layer, paddle.nn.Embedding)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Unfold(kernel_sizes=2)
    assert_type(layer, paddle.nn.Unfold)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Fold(output_sizes=[2, 2], kernel_sizes=2)
    assert_type(layer, paddle.nn.Fold)
    assert_type(layer(tensor), Tensor)
