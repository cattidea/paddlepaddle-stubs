# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing_extensions import assert_type

import paddle


def test_import():
    paddle.vision.models.ResNet
    paddle.vision.models.resnet18
    paddle.vision.models.resnet34
    paddle.vision.models.resnet50
    paddle.vision.models.resnet101
    paddle.vision.models.resnet152
    paddle.vision.models.resnext50_32x4d
    paddle.vision.models.resnext50_64x4d
    paddle.vision.models.resnext101_32x4d
    paddle.vision.models.resnext101_64x4d
    paddle.vision.models.resnext152_32x4d
    paddle.vision.models.resnext152_64x4d
    paddle.vision.models.wide_resnet50_2
    paddle.vision.models.wide_resnet101_2

    paddle.vision.ResNet
    paddle.vision.resnet18
    paddle.vision.resnet34
    paddle.vision.resnet50
    paddle.vision.resnet101
    paddle.vision.resnet152
    paddle.vision.resnext50_32x4d
    paddle.vision.resnext50_64x4d
    paddle.vision.resnext101_32x4d
    paddle.vision.resnext101_64x4d
    paddle.vision.resnext152_32x4d
    paddle.vision.resnext152_64x4d
    paddle.vision.wide_resnet50_2
    paddle.vision.wide_resnet101_2

    from paddle.vision import (
        ResNet,  # pyright: ignore[reportUnusedImport]
        resnet18,  # pyright: ignore[reportUnusedImport]
        resnet34,  # pyright: ignore[reportUnusedImport]
        resnet50,  # pyright: ignore[reportUnusedImport]
        resnet101,  # pyright: ignore[reportUnusedImport]
        resnet152,  # pyright: ignore[reportUnusedImport]
        resnext50_32x4d,  # pyright: ignore[reportUnusedImport]
        resnext50_64x4d,  # pyright: ignore[reportUnusedImport]
        resnext101_32x4d,  # pyright: ignore[reportUnusedImport]
        resnext101_64x4d,  # pyright: ignore[reportUnusedImport]
        resnext152_32x4d,  # pyright: ignore[reportUnusedImport]
        resnext152_64x4d,  # pyright: ignore[reportUnusedImport]
        wide_resnet50_2,  # pyright: ignore[reportUnusedImport]
        wide_resnet101_2,  # pyright: ignore[reportUnusedImport]
    )
    from paddle.vision.models import (
        ResNet,  # pyright: ignore[reportUnusedImport]
        resnet18,  # pyright: ignore[reportUnusedImport]
        resnet34,  # pyright: ignore[reportUnusedImport]
        resnet50,  # pyright: ignore[reportUnusedImport]
        resnet101,  # pyright: ignore[reportUnusedImport]
        resnet152,  # pyright: ignore[reportUnusedImport]
        resnext50_32x4d,  # pyright: ignore[reportUnusedImport]
        resnext50_64x4d,  # pyright: ignore[reportUnusedImport]
        resnext101_32x4d,  # pyright: ignore[reportUnusedImport]
        resnext101_64x4d,  # pyright: ignore[reportUnusedImport]
        resnext152_32x4d,  # pyright: ignore[reportUnusedImport]
        resnext152_64x4d,  # pyright: ignore[reportUnusedImport]
        wide_resnet50_2,  # pyright: ignore[reportUnusedImport]
        wide_resnet101_2,  # pyright: ignore[reportUnusedImport]
    )


def test_creation():
    model = paddle.vision.models.resnet18(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ResNet)
    model = paddle.vision.models.resnet34(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ResNet)
    model = paddle.vision.models.resnet50(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ResNet)
    model = paddle.vision.models.resnet101(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ResNet)
    model = paddle.vision.models.resnet152(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ResNet)
    model = paddle.vision.models.resnext50_32x4d(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ResNet)
    model = paddle.vision.models.resnext50_64x4d(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ResNet)
    model = paddle.vision.models.resnext101_32x4d(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ResNet)
    model = paddle.vision.models.resnext101_64x4d(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ResNet)
    model = paddle.vision.models.resnext152_32x4d(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ResNet)
    model = paddle.vision.models.resnext152_64x4d(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ResNet)
    model = paddle.vision.models.wide_resnet50_2(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ResNet)
    model = paddle.vision.models.wide_resnet101_2(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ResNet)


def test_forward():
    model = paddle.vision.models.resnet18(pretrained=False, num_classes=10, with_pool=True)
    x = paddle.randn([1, 3, 224, 224])
    out = model(x)
    assert_type(out, paddle.Tensor)
