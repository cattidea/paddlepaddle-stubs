# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from paddle import Tensor
from typing_extensions import assert_type


def test_import():
    paddle.Tensor

    from paddle import Tensor  # pyright: ignore [reportUnusedImport]


def test_creation():
    tensor = paddle.to_tensor([1])
    assert_type(tensor, Tensor)


def test_methods():
    a = paddle.to_tensor([1])
    b = paddle.to_tensor([2])

    c = a + b
    d = a - b
    e = a * b
    f = a / b
    g = a**b
    h = a // b
    i = a % b
    j = len(a)
    for k in iter(a):
        assert_type(k, Tensor)
    l = a[1]
    m = a[:1]
    n = a[b]

    # TODO: more methods
