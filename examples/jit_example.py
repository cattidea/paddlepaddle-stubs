from __future__ import annotations


import paddle

from paddle import Tensor


def fn(x: Tensor) -> Tensor:
    return x + 1


x = paddle.to_tensor(1)
static_fn = paddle.jit.to_static(fn)
y = static_fn(x)

class MyModel(paddle.nn.Layer):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x + 1

model = MyModel()
static_model = paddle.jit.to_static(model)
y = static_model(x)

