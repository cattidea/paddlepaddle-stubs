# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing_extensions import assert_type

import paddle


def test_import():
    paddle.vision.models.DenseNet
    paddle.vision.models.densenet121
    paddle.vision.models.densenet161
    paddle.vision.models.densenet169
    paddle.vision.models.densenet201
    paddle.vision.models.densenet264

    paddle.vision.DenseNet
    paddle.vision.densenet121
    paddle.vision.densenet161
    paddle.vision.densenet169
    paddle.vision.densenet201
    paddle.vision.densenet264

    from paddle.vision import (
        DenseNet,  # pyright: ignore[reportUnusedImport]
        densenet121,  # pyright: ignore[reportUnusedImport]
        densenet161,  # pyright: ignore[reportUnusedImport]
        densenet169,  # pyright: ignore[reportUnusedImport]
        densenet201,  # pyright: ignore[reportUnusedImport]
        densenet264,  # pyright: ignore[reportUnusedImport]
    )
    from paddle.vision.models import (
        DenseNet,  # pyright: ignore[reportUnusedImport]
        densenet121,  # pyright: ignore[reportUnusedImport]
        densenet161,  # pyright: ignore[reportUnusedImport]
        densenet169,  # pyright: ignore[reportUnusedImport]
        densenet201,  # pyright: ignore[reportUnusedImport]
        densenet264,  # pyright: ignore[reportUnusedImport]
    )


def test_creation():
    model = paddle.vision.models.densenet121(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.DenseNet)
    model = paddle.vision.models.densenet161(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.DenseNet)
    model = paddle.vision.models.densenet169(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.DenseNet)
    model = paddle.vision.models.densenet201(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.DenseNet)
    model = paddle.vision.models.densenet264(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.DenseNet)


def test_forward():
    model = paddle.vision.models.densenet121(pretrained=False, num_classes=10, with_pool=True)
    x = paddle.randn([1, 3, 224, 224])
    out = model(x)
    assert_type(out, paddle.Tensor)
