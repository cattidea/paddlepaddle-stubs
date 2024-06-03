# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing_extensions import assert_type

import paddle


def test_import():
    paddle.vision.models.LeNet

    paddle.vision.LeNet

    from paddle.vision import LeNet  # pyright: ignore[reportUnusedImport]
    from paddle.vision.models import LeNet  # pyright: ignore[reportUnusedImport]


def test_creation():
    model = paddle.vision.models.LeNet(num_classes=10)
    assert_type(model, paddle.vision.models.LeNet)


def test_forward():
    model = paddle.vision.models.LeNet(num_classes=10)
    x = paddle.randn([1, 1, 32, 32])
    out = model(x)
    assert_type(out, paddle.Tensor)
