# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from paddle import Tensor
from typing_extensions import assert_type


def test_import():
    paddle.nn.CELU
    paddle.nn.ELU
    paddle.nn.GELU
    paddle.nn.Hardshrink
    paddle.nn.Hardswish
    paddle.nn.Tanh
    paddle.nn.Hardtanh
    paddle.nn.PReLU
    paddle.nn.ReLU
    paddle.nn.ReLU6
    paddle.nn.SELU
    paddle.nn.LeakyReLU
    paddle.nn.Sigmoid
    paddle.nn.Hardsigmoid
    paddle.nn.Softplus
    paddle.nn.Softshrink
    paddle.nn.Softsign
    paddle.nn.Swish
    paddle.nn.Mish
    paddle.nn.Tanhshrink
    paddle.nn.ThresholdedReLU
    paddle.nn.Silu
    paddle.nn.LogSigmoid
    paddle.nn.Softmax
    paddle.nn.LogSigmoid
    paddle.nn.Maxout

    from paddle.nn import CELU  # pyright: ignore [reportUnusedImport]
    from paddle.nn import ELU  # pyright: ignore [reportUnusedImport]
    from paddle.nn import GELU  # pyright: ignore [reportUnusedImport]
    from paddle.nn import SELU  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Hardshrink  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Hardsigmoid  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Hardswish  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Hardtanh  # pyright: ignore [reportUnusedImport]
    from paddle.nn import LeakyReLU  # pyright: ignore [reportUnusedImport]
    from paddle.nn import LogSigmoid  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Maxout  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Mish  # pyright: ignore [reportUnusedImport]
    from paddle.nn import PReLU  # pyright: ignore [reportUnusedImport]
    from paddle.nn import ReLU  # pyright: ignore [reportUnusedImport]
    from paddle.nn import ReLU6  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Sigmoid  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Silu  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Softmax  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Softplus  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Softshrink  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Softsign  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Swish  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Tanh  # pyright: ignore [reportUnusedImport]
    from paddle.nn import Tanhshrink  # pyright: ignore [reportUnusedImport]
    from paddle.nn import ThresholdedReLU  # pyright: ignore [reportUnusedImport]


def test_types():
    tensor = paddle.to_tensor([1, 2, 3])

    layer = paddle.nn.CELU()
    assert_type(layer, paddle.nn.CELU)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.ELU()
    assert_type(layer, paddle.nn.ELU)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.GELU()
    assert_type(layer, paddle.nn.GELU)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Hardshrink()
    assert_type(layer, paddle.nn.Hardshrink)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Hardswish()
    assert_type(layer, paddle.nn.Hardswish)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Tanh()
    assert_type(layer, paddle.nn.Tanh)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Hardtanh()
    assert_type(layer, paddle.nn.Hardtanh)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.PReLU()
    assert_type(layer, paddle.nn.PReLU)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.ReLU()
    assert_type(layer, paddle.nn.ReLU)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.ReLU6()
    assert_type(layer, paddle.nn.ReLU6)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.SELU()
    assert_type(layer, paddle.nn.SELU)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.LeakyReLU()
    assert_type(layer, paddle.nn.LeakyReLU)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Sigmoid()
    assert_type(layer, paddle.nn.Sigmoid)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Hardsigmoid()
    assert_type(layer, paddle.nn.Hardsigmoid)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Softplus()
    assert_type(layer, paddle.nn.Softplus)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Softshrink()
    assert_type(layer, paddle.nn.Softshrink)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Softsign()
    assert_type(layer, paddle.nn.Softsign)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Swish()
    assert_type(layer, paddle.nn.Swish)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Mish()
    assert_type(layer, paddle.nn.Mish)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Tanhshrink()
    assert_type(layer, paddle.nn.Tanhshrink)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.ThresholdedReLU()
    assert_type(layer, paddle.nn.ThresholdedReLU)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Silu()
    assert_type(layer, paddle.nn.Silu)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.LogSigmoid()
    assert_type(layer, paddle.nn.LogSigmoid)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Softmax()
    assert_type(layer, paddle.nn.Softmax)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.LogSigmoid()
    assert_type(layer, paddle.nn.LogSigmoid)
    assert_type(layer(tensor), Tensor)

    layer = paddle.nn.Maxout(groups=4)
    assert_type(layer, paddle.nn.Maxout)
    assert_type(layer(tensor), Tensor)
