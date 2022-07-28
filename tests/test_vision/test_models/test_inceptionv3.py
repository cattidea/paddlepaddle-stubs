# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from typing_extensions import assert_type


def test_import():
    paddle.vision.models.InceptionV3
    paddle.vision.models.inception_v3

    paddle.vision.InceptionV3
    paddle.vision.inception_v3

    from paddle.vision import InceptionV3  # pyright: strict, reportUnusedImport=false
    from paddle.vision import inception_v3  # pyright: strict, reportUnusedImport=false
    from paddle.vision.models import (
        InceptionV3,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        inception_v3,  # pyright: strict, reportUnusedImport=false
    )


def test_creation():
    model = paddle.vision.models.inception_v3(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.InceptionV3)


def test_forward():
    model = paddle.vision.models.inception_v3(pretrained=False, num_classes=10, with_pool=True)
    x = paddle.randn([1, 3, 299, 299])
    out = model(x)
    assert_type(out, paddle.Tensor)
