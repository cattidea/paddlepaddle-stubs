# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing_extensions import assert_type

import paddle


def test_import():
    paddle.vision.models.MobileNetV3Small
    paddle.vision.models.MobileNetV3Large
    paddle.vision.models.mobilenet_v3_small
    paddle.vision.models.mobilenet_v3_large

    paddle.vision.MobileNetV3Small
    paddle.vision.MobileNetV3Large
    paddle.vision.mobilenet_v3_small
    paddle.vision.mobilenet_v3_large

    from paddle.vision import (
        MobileNetV3Large,  # pyright: ignore[reportUnusedImport]
        MobileNetV3Small,  # pyright: ignore[reportUnusedImport]
        mobilenet_v3_large,  # pyright: ignore[reportUnusedImport]
        mobilenet_v3_small,  # pyright: ignore[reportUnusedImport]
    )
    from paddle.vision.models import (
        MobileNetV3Large,  # pyright: ignore[reportUnusedImport]
        MobileNetV3Small,  # pyright: ignore[reportUnusedImport]
        mobilenet_v3_large,  # pyright: ignore[reportUnusedImport]
        mobilenet_v3_small,  # pyright: ignore[reportUnusedImport]
    )


def test_creation():
    model = paddle.vision.models.mobilenet_v3_small(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.MobileNetV3Small)
    model = paddle.vision.models.mobilenet_v3_large(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.MobileNetV3Large)


def test_forward():
    model = paddle.vision.models.mobilenet_v3_small(pretrained=False, num_classes=10, with_pool=True)
    x = paddle.randn([1, 3, 224, 224])
    out = model(x)
    assert_type(out, paddle.Tensor)

    model = paddle.vision.models.mobilenet_v3_large(pretrained=False, num_classes=10, with_pool=True)
    x = paddle.randn([1, 3, 224, 224])
    out = model(x)
    assert_type(out, paddle.Tensor)
