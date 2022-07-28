# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from typing_extensions import assert_type


def test_import():
    paddle.vision.models.MobileNetV1
    paddle.vision.models.mobilenet_v1

    paddle.vision.MobileNetV1
    paddle.vision.mobilenet_v1

    from paddle.vision import MobileNetV1  # pyright: strict, reportUnusedImport=false
    from paddle.vision import mobilenet_v1  # pyright: strict, reportUnusedImport=false
    from paddle.vision.models import (
        MobileNetV1,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        mobilenet_v1,  # pyright: strict, reportUnusedImport=false
    )


def test_creation():
    model = paddle.vision.models.mobilenet_v1(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.MobileNetV1)


def test_forward():
    model = paddle.vision.models.mobilenet_v1(pretrained=False, num_classes=10, with_pool=True)
    x = paddle.randn([1, 3, 224, 224])
    out = model(x)
    assert_type(out, paddle.Tensor)
