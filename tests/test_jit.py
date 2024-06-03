# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing_extensions import assert_type

import paddle


def test_import():
    paddle.jit.to_static

    from paddle.jit import (
        to_static,  # pyright: ignore [reportUnusedImport]
    )


def test_static_net_without_params_1():
    class Net(paddle.nn.Layer):
        @paddle.jit.to_static
        def forward(self, x: paddle.Tensor) -> paddle.Tensor:  # type: ignore
            return x

    net = Net()
    assert_type(net, Net)


def test_static_net_without_params_2():
    class Net(paddle.nn.Layer):
        def forward(self, x: paddle.Tensor) -> paddle.Tensor:  # type: ignore
            return x

    net = Net()
    static_net = paddle.jit.to_static(net)
    assert_type(static_net, Net)


def test_static_net_with_params_1():
    class Net(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.fc = paddle.nn.Linear(10, 10)

        def forward(self, x: paddle.Tensor) -> paddle.Tensor:  # type: ignore
            return self.fc(x)

    net = Net()
    static_net = paddle.jit.to_static(net, input_spec=[paddle.static.InputSpec(shape=[None, 10])])
    assert_type(static_net, Net)


def test_static_net_with_params_2():
    class Net(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.fc = paddle.nn.Linear(10, 10)

        @paddle.jit.to_static(input_spec=[paddle.static.InputSpec(shape=[None, 10])])
        def forward(self, x: paddle.Tensor) -> paddle.Tensor:  # type: ignore
            return self.fc(x)

    net = Net()
    assert_type(net, Net)


def test_static_net_with_params_3():
    class Net(paddle.nn.Layer):
        def __init__(self):
            super().__init__()
            self.fc = paddle.nn.Linear(10, 10)

        def forward(self, x: paddle.Tensor) -> paddle.Tensor:  # type: ignore
            return self.fc(x)

    net = Net()
    static_net = paddle.jit.to_static(input_spec=[paddle.static.InputSpec(shape=[None, 10])])(net)
    assert_type(static_net, Net)


def test_static_fn_without_params_1():
    @paddle.jit.to_static
    def fn(x: paddle.Tensor) -> paddle.Tensor:  # type: ignore
        return x

    x = paddle.randn([10, 10])
    y = fn(x)
    assert_type(y, paddle.Tensor)


def test_static_fn_without_params_2():
    def fn(x: paddle.Tensor) -> paddle.Tensor:  # type: ignore
        return x

    static_fn = paddle.jit.to_static(fn)
    x = paddle.randn([10, 10])
    y = static_fn(x)
    assert_type(y, paddle.Tensor)


def test_static_fn_with_params_1():
    @paddle.jit.to_static(input_spec=[paddle.static.InputSpec(shape=[None, 10])])
    def fn(x: paddle.Tensor) -> paddle.Tensor:  # type: ignore
        return x

    x = paddle.randn([10, 10])
    y = fn(x)
    assert_type(y, paddle.Tensor)


def test_static_fn_with_params_2():
    def fn(x: paddle.Tensor) -> paddle.Tensor:  # type: ignore
        return x

    static_fn = paddle.jit.to_static(input_spec=[paddle.static.InputSpec(shape=[None, 10])])(fn)
    x = paddle.randn([10, 10])
    y = static_fn(x)
    assert_type(y, paddle.Tensor)


def test_static_fn_with_params_3():
    def fn(x: paddle.Tensor) -> paddle.Tensor:  # type: ignore
        return x

    static_fn = paddle.jit.to_static(fn, input_spec=[paddle.static.InputSpec(shape=[None, 10])])
    x = paddle.randn([10, 10])
    y = static_fn(x)
    assert_type(y, paddle.Tensor)
