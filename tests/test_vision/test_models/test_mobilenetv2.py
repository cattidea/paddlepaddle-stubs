# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from typing_extensions import assert_type


def test_import():
    paddle.vision.models.MobileNetV2
    paddle.vision.models.mobilenet_v2

    paddle.vision.MobileNetV2
    paddle.vision.mobilenet_v2

    from paddle.vision import MobileNetV2  # pyright: ignore[reportUnusedImport]
    from paddle.vision import mobilenet_v2  # pyright: ignore[reportUnusedImport]
    from paddle.vision.models import MobileNetV2  # pyright: ignore[reportUnusedImport]
    from paddle.vision.models import mobilenet_v2  # pyright: ignore[reportUnusedImport]


def test_creation():
    model = paddle.vision.models.mobilenet_v2(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.MobileNetV2)


def test_forward():
    model = paddle.vision.models.mobilenet_v2(pretrained=False, num_classes=10, with_pool=True)
    x = paddle.randn([1, 3, 224, 224])
    out = model(x)
    assert_type(out, paddle.Tensor)
