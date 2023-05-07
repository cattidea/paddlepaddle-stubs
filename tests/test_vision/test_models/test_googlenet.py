# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from typing_extensions import assert_type


def test_import():
    paddle.vision.models.GoogLeNet
    paddle.vision.models.googlenet

    paddle.vision.GoogLeNet
    paddle.vision.googlenet

    from paddle.vision import GoogLeNet  # pyright: ignore[reportUnusedImport]
    from paddle.vision import googlenet  # pyright: ignore[reportUnusedImport]
    from paddle.vision.models import GoogLeNet  # pyright: ignore[reportUnusedImport]
    from paddle.vision.models import googlenet  # pyright: ignore[reportUnusedImport]


def test_creation():
    model = paddle.vision.models.googlenet(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.GoogLeNet)


def test_forward():
    model = paddle.vision.models.googlenet(pretrained=False, num_classes=10, with_pool=True)
    x = paddle.randn([1, 3, 224, 224])
    out, out1, out2 = model(x)
    assert_type(out, paddle.Tensor)
    assert_type(out1, paddle.Tensor)
    assert_type(out2, paddle.Tensor)
