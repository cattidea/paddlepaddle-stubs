# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from paddle import Tensor
from typing_extensions import assert_type


def test_import():
    paddle.nn.BatchNorm
    paddle.nn.SpectralNorm
    paddle.nn.InstanceNorm1D
    paddle.nn.InstanceNorm2D
    paddle.nn.InstanceNorm3D
    paddle.nn.GroupNorm
    paddle.nn.LayerNorm
    paddle.nn.BatchNorm1D
    paddle.nn.BatchNorm2D
    paddle.nn.BatchNorm3D
    paddle.nn.SyncBatchNorm
    paddle.nn.LocalResponseNorm

    from paddle.nn import BatchNorm  # pyright: ignore [reportUnusedImport]
    from paddle.nn import BatchNorm1D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import BatchNorm2D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import BatchNorm3D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import GroupNorm  # pyright: ignore [reportUnusedImport]
    from paddle.nn import InstanceNorm1D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import InstanceNorm2D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import InstanceNorm3D  # pyright: ignore [reportUnusedImport]
    from paddle.nn import LayerNorm  # pyright: ignore [reportUnusedImport]
    from paddle.nn import LocalResponseNorm  # pyright: ignore [reportUnusedImport]
    from paddle.nn import SpectralNorm  # pyright: ignore [reportUnusedImport]
    from paddle.nn import SyncBatchNorm  # pyright: ignore [reportUnusedImport]


def test_types():
    tensor = paddle.randint(0, 255, shape=[3, 224, 224])

    norm = paddle.nn.BatchNorm(3)
    assert_type(norm, paddle.nn.BatchNorm)
    assert_type(norm(tensor), Tensor)

    norm = paddle.nn.BatchNorm1D(3)
    assert_type(norm, paddle.nn.BatchNorm1D)
    assert_type(norm(tensor), Tensor)

    norm = paddle.nn.BatchNorm2D(3)
    assert_type(norm, paddle.nn.BatchNorm2D)
    assert_type(norm(tensor), Tensor)

    norm = paddle.nn.BatchNorm3D(3)
    assert_type(norm, paddle.nn.BatchNorm3D)
    assert_type(norm(tensor), Tensor)

    norm = paddle.nn.GroupNorm(3, 3)
    assert_type(norm, paddle.nn.GroupNorm)
    assert_type(norm(tensor), Tensor)

    norm = paddle.nn.InstanceNorm1D(3)
    assert_type(norm, paddle.nn.InstanceNorm1D)
    assert_type(norm(tensor), Tensor)

    norm = paddle.nn.InstanceNorm2D(3)
    assert_type(norm, paddle.nn.InstanceNorm2D)
    assert_type(norm(tensor), Tensor)

    norm = paddle.nn.InstanceNorm3D(3)
    assert_type(norm, paddle.nn.InstanceNorm3D)
    assert_type(norm(tensor), Tensor)

    norm = paddle.nn.LayerNorm(3)
    assert_type(norm, paddle.nn.LayerNorm)
    assert_type(norm(tensor), Tensor)

    norm = paddle.nn.LocalResponseNorm(3)
    assert_type(norm, paddle.nn.LocalResponseNorm)
    assert_type(norm(tensor), Tensor)

    norm = paddle.nn.SpectralNorm([3])
    assert_type(norm, paddle.nn.SpectralNorm)
    assert_type(norm(tensor), Tensor)
