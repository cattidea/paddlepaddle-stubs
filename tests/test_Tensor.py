# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import numpy as np
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
    c = a - b
    c = a * b
    c = a / b
    c = -a
    c = a // b
    c = a % b
    c = a**b
    c = a @ b

    c = a == b
    c = a != b
    c = a <= b
    c = a >= b
    c = a < b
    c = a > b

    c = a & b
    c = a | b
    c = a ^ b
    c = ~a

    c = repr(a)
    c = str(a)
    c = len(a)
    for k in iter(a):
        assert_type(k, Tensor)
    c = a[0]
    c = a[:1]
    c = a[b]
    c = float(a)
    c = int(a)
    c = bool(a)
    c = a.__index__()
    c = hash(a)
    c = a in b

    # attributes
    c = a.dtype
    c = a.shape
    c = a.grad
    c = a.is_leaf
    c = a.name
    c = a.ndim
    c = a.persistable
    c = a.place
    c = a.stop_gradient
    c = a.T
    c = a.__array_ufunc__

    # methods
    a.clear_grad()
    a.clear_gradient()
    a.item()
    a.item(1, 2)
    a.astype(paddle.float32)
    a.astype("float32")
    a.astype(np.float32)
    a.reshape([1, 2, 3])
    a.reshape((1,))
    a.reshape(a)
    a.set_value([1])
    c = a.numpy()
    a.backward()
    c = a.clone()
    c = a.broadcast_to([1, 2])
    c = a.cast(paddle.float32)
    c = a.size()
    c = a.dim()
    c = a.ndimension()
    c = a.imag()
    c = a.real()

    # TODO: more methods
