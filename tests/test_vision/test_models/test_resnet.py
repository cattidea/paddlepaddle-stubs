# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from typing_extensions import assert_type


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

    from paddle.vision import ResNet  # pyright: strict, reportUnusedImport=false
    from paddle.vision import resnet18  # pyright: strict, reportUnusedImport=false
    from paddle.vision import resnet34  # pyright: strict, reportUnusedImport=false
    from paddle.vision import resnet50  # pyright: strict, reportUnusedImport=false
    from paddle.vision import resnet101  # pyright: strict, reportUnusedImport=false
    from paddle.vision import resnet152  # pyright: strict, reportUnusedImport=false
    from paddle.vision import (
        resnext50_32x4d,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision import (
        resnext50_64x4d,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision import (
        resnext101_32x4d,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision import (
        resnext101_64x4d,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision import (
        resnext152_32x4d,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision import (
        resnext152_64x4d,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision import (
        wide_resnet50_2,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision import (
        wide_resnet101_2,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import ResNet  # pyright: strict, reportUnusedImport=false
    from paddle.vision.models import (
        resnet18,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        resnet34,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        resnet50,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        resnet101,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        resnet152,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        resnext50_32x4d,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        resnext50_64x4d,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        resnext101_32x4d,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        resnext101_64x4d,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        resnext152_32x4d,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        resnext152_64x4d,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        wide_resnet50_2,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        wide_resnet101_2,  # pyright: strict, reportUnusedImport=false
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
