# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from typing_extensions import assert_type


def test_import():
    paddle.vision.models.ShuffleNetV2
    paddle.vision.models.shufflenet_v2_x0_25
    paddle.vision.models.shufflenet_v2_x0_5
    paddle.vision.models.shufflenet_v2_x1_0
    paddle.vision.models.shufflenet_v2_x1_5
    paddle.vision.models.shufflenet_v2_x2_0
    paddle.vision.models.shufflenet_v2_swish

    paddle.vision.ShuffleNetV2
    paddle.vision.shufflenet_v2_x0_25
    paddle.vision.shufflenet_v2_x0_5
    paddle.vision.shufflenet_v2_x1_0
    paddle.vision.shufflenet_v2_x1_5
    paddle.vision.shufflenet_v2_x2_0
    paddle.vision.shufflenet_v2_swish

    from paddle.vision import ShuffleNetV2  # pyright: strict, reportUnusedImport=false
    from paddle.vision import (
        shufflenet_v2_swish,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision import (
        shufflenet_v2_x0_5,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision import (
        shufflenet_v2_x0_25,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision import (
        shufflenet_v2_x1_0,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision import (
        shufflenet_v2_x1_5,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision import (
        shufflenet_v2_x2_0,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        ShuffleNetV2,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        shufflenet_v2_swish,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        shufflenet_v2_x0_5,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        shufflenet_v2_x0_25,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        shufflenet_v2_x1_0,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        shufflenet_v2_x1_5,  # pyright: strict, reportUnusedImport=false
    )
    from paddle.vision.models import (
        shufflenet_v2_x2_0,  # pyright: strict, reportUnusedImport=false
    )


def test_creation():
    model = paddle.vision.models.shufflenet_v2_x0_25(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ShuffleNetV2)
    model = paddle.vision.models.shufflenet_v2_x0_5(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ShuffleNetV2)
    model = paddle.vision.models.shufflenet_v2_x1_0(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ShuffleNetV2)
    model = paddle.vision.models.shufflenet_v2_x1_5(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ShuffleNetV2)
    model = paddle.vision.models.shufflenet_v2_x2_0(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ShuffleNetV2)
    model = paddle.vision.models.shufflenet_v2_swish(pretrained=False, num_classes=10, with_pool=True)
    assert_type(model, paddle.vision.models.ShuffleNetV2)


def test_forward():
    model = paddle.vision.models.shufflenet_v2_x0_25(pretrained=False, num_classes=10, with_pool=True)
    x = paddle.randn([1, 3, 224, 224])
    out = model(x)
    assert_type(out, paddle.Tensor)
