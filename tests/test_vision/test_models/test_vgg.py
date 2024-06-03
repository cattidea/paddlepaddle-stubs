# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing_extensions import assert_type

import paddle


def test_import():
    paddle.vision.models.VGG
    paddle.vision.models.vgg11
    paddle.vision.models.vgg13
    paddle.vision.models.vgg16
    paddle.vision.models.vgg19

    paddle.vision.VGG
    paddle.vision.vgg11
    paddle.vision.vgg13
    paddle.vision.vgg16
    paddle.vision.vgg19

    from paddle.vision import (
        VGG,  # pyright: ignore[reportUnusedImport]
        vgg11,  # pyright: ignore[reportUnusedImport]
        vgg13,  # pyright: ignore[reportUnusedImport]
        vgg16,  # pyright: ignore[reportUnusedImport]
        vgg19,  # pyright: ignore[reportUnusedImport]
    )
    from paddle.vision.models import (
        VGG,  # pyright: ignore[reportUnusedImport]
        vgg11,  # pyright: ignore[reportUnusedImport]
        vgg13,  # pyright: ignore[reportUnusedImport]
        vgg16,  # pyright: ignore[reportUnusedImport]
        vgg19,  # pyright: ignore[reportUnusedImport]
    )


def test_creation():
    model = paddle.vision.models.vgg11(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.VGG)
    model = paddle.vision.models.vgg13(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.VGG)
    model = paddle.vision.models.vgg16(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.VGG)
    model = paddle.vision.models.vgg19(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.VGG)


def test_forward():
    model = paddle.vision.models.vgg11(pretrained=False, num_classes=10, with_pool=True)
    x = paddle.randn([1, 3, 224, 224])
    out = model(x)
    assert_type(out, paddle.Tensor)
