# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing_extensions import assert_type

import paddle
from paddle import Tensor, nn


def test_import():
    paddle.nn.Layer

    from paddle.nn import Layer  # pyright: ignore [reportUnusedImport]


def test_creation():
    layer = paddle.nn.Layer("name", "float32")
    layer = paddle.nn.Layer("name", paddle.float32)
    assert_type(layer, paddle.nn.Layer)


def test_methods():
    layer = paddle.nn.Layer("name", "float32")

    # Children
    for layer in layer.children():
        assert_type(layer, paddle.nn.Layer)

    for name, layer in layer.named_children():
        assert_type(layer, paddle.nn.Layer)
        assert_type(name, str)

    # Sublayers
    layer.add_sublayer("name", layer)
    for layer in layer.sublayers():
        assert_type(layer, paddle.nn.Layer)

    for name, layer in layer.named_sublayers():
        assert_type(layer, paddle.nn.Layer)
        assert_type(name, str)

    # Parameters
    assert_type(
        layer.create_parameter(
            shape=[1, 1],
            attr=paddle.ParamAttr(),
            dtype="float32",
            is_bias=False,
            default_initializer=paddle.nn.initializer.Constant(0.5),
        ),
        Tensor,
    )

    assert_type(
        layer.add_parameter(name="name", parameter=paddle.to_tensor([1, 1], dtype="float32")),
        Tensor,
    )

    for parameter in layer.parameters():
        assert_type(parameter, Tensor)

    for name, parameter in layer.named_parameters():
        assert_type(parameter, Tensor)
        assert_type(name, str)

    # Tensor
    assert_type(
        layer.create_tensor(name="tensor", persistable=False, dtype="float32"),
        Tensor,
    )

    # Buffer
    layer.register_buffer("buf", paddle.to_tensor([1, 2]), True)

    for buffer in layer.buffers():
        assert_type(buffer, Tensor)

    for name, buffer in layer.named_buffers():
        assert_type(buffer, Tensor)
        assert_type(name, str)

    # Training process
    layer(paddle.to_tensor([1, 2]))

    layer.train()
    layer.eval()
    layer.forward(paddle.to_tensor([1, 2]))
    layer.backward(paddle.to_tensor([1, 2]))
    layer.clear_gradients()

    # Hooks
    def _forward_pre_hook(layer: nn.Layer, input: Tensor) -> Tensor:
        return input * 2

    def _forward_post_hook(layer: nn.Layer, input: Tensor, output: Tensor) -> Tensor:
        return output * 2

    layer.register_forward_pre_hook(_forward_pre_hook)
    layer.register_forward_post_hook(_forward_post_hook)

    # Magic methods related
    layer.any_attr_blablabla
    layer.any_attr_blablabla = "blablabla"
    del layer.any_attr_blablabla
    for attr in dir(layer):
        assert_type(attr, str)
    assert_type(layer.extra_repr(), str)

    # State dict
    for key, value in layer.to_static_state_dict().items():
        assert_type(key, str)
        assert_type(value, Tensor)

    for key, value in layer.state_dict().items():
        assert_type(key, str)
        assert_type(value, Tensor)

    layer.set_state_dict(
        {
            "weight": paddle.to_tensor([1, 2], dtype="float32"),
            "bias": paddle.to_tensor([1, 2], dtype="float32"),
        }
    )
    # alias
    layer.load_dict(layer.state_dict())
    layer.set_dict(layer.state_dict())

    # Others
    layer.to("cpu")
    layer.to("cuda:0")

    def _init_weights(layer: paddle.nn.Layer):
        if isinstance(layer, paddle.nn.Linear):
            new_weight = paddle.full(shape=layer.weight.shape, dtype=layer.weight.dtype, fill_value=0.9)
            layer.weight.set_value(new_weight)

    layer.apply(_init_weights)
    assert_type(layer.full_name(), str)
