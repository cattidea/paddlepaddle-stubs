# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing_extensions import assert_type

import paddle


def test_import():
    paddle.vision.models.AlexNet
    paddle.vision.models.alexnet

    paddle.vision.AlexNet
    paddle.vision.alexnet

    from paddle.vision import (
        AlexNet,  # pyright: ignore[reportUnusedImport]
        alexnet,  # pyright: ignore[reportUnusedImport]
    )
    from paddle.vision.models import (
        AlexNet,  # pyright: ignore[reportUnusedImport]
        alexnet,  # pyright: ignore[reportUnusedImport]
    )


def test_creation():
    model = paddle.vision.models.alexnet(pretrained=False)
    assert_type(model, paddle.vision.models.AlexNet)


def test_forward():
    model = paddle.vision.models.alexnet(pretrained=False)
    x = paddle.randn([1, 3, 224, 224])
    out = model(x)
    assert_type(out, paddle.Tensor)
