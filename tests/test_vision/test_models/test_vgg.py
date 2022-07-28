# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from typing_extensions import assert_type


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

    from paddle.vision import VGG  # pyright: strict, reportUnusedImport=false
    from paddle.vision import vgg11  # pyright: strict, reportUnusedImport=false
    from paddle.vision import vgg13  # pyright: strict, reportUnusedImport=false
    from paddle.vision import vgg16  # pyright: strict, reportUnusedImport=false
    from paddle.vision import vgg19  # pyright: strict, reportUnusedImport=false
    from paddle.vision.models import VGG  # pyright: strict, reportUnusedImport=false
    from paddle.vision.models import vgg11  # pyright: strict, reportUnusedImport=false
    from paddle.vision.models import vgg13  # pyright: strict, reportUnusedImport=false
    from paddle.vision.models import vgg16  # pyright: strict, reportUnusedImport=false
    from paddle.vision.models import vgg19  # pyright: strict, reportUnusedImport=false


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
