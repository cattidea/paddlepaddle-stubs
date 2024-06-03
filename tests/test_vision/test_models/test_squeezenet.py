# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing_extensions import assert_type

import paddle


def test_import():
    paddle.vision.models.SqueezeNet
    paddle.vision.models.squeezenet1_0
    paddle.vision.models.squeezenet1_1

    paddle.vision.SqueezeNet
    paddle.vision.squeezenet1_0
    paddle.vision.squeezenet1_1

    from paddle.vision import (
        SqueezeNet,  # pyright: ignore[reportUnusedImport]
        squeezenet1_0,  # pyright: ignore[reportUnusedImport]
        squeezenet1_1,  # pyright: ignore[reportUnusedImport]
    )
    from paddle.vision.models import (
        SqueezeNet,  # pyright: ignore[reportUnusedImport]
        squeezenet1_0,  # pyright: ignore[reportUnusedImport]
        squeezenet1_1,  # pyright: ignore[reportUnusedImport]
    )


def test_creation():
    model = paddle.vision.models.squeezenet1_0(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.SqueezeNet)
    model = paddle.vision.models.squeezenet1_1(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.SqueezeNet)


def test_forward():
    model = paddle.vision.models.squeezenet1_0(pretrained=False, num_classes=10, with_pool=True)
    x = paddle.randn([1, 3, 224, 224])
    out = model(x)
    assert_type(out, paddle.Tensor)
