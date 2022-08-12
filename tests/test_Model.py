# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from paddle.static import InputSpec
from typing_extensions import assert_type


def test_import():
    paddle.Model

    from paddle import Model  # pyright: ignore [reportUnusedImport]


def test_creation():
    layer = paddle.nn.Layer("name", "float32")
    model = paddle.Model(layer, InputSpec(shape=[1, 2], dtype="float32"))
    assert_type(model, paddle.Model)


def test_methods():
    input_spec = InputSpec([None, 784], "float32", "x")
    label_spec = InputSpec([None, 1], "int64", "label")

    layer = paddle.nn.Layer("name", "float32")
    model = paddle.Model(layer, input_spec, label_spec)

    optim = paddle.optimizer.SGD(learning_rate=1e-3, parameters=model.parameters())
    model.prepare(optim, paddle.nn.CrossEntropyLoss(), metrics=paddle.metric.Accuracy())

    data = paddle.rand((4, 784), dtype="float32")
    label = paddle.randint(0, 10, (4, 1), dtype="int64")
    loss = model.train_batch([data], [label])
    loss, acc = model.eval_batch([data], [label])

    pred_out = model.predict_batch([data])

    model.save("checkpoint/test")
    model.load("checkpoint/test")

    params = model.parameters()
    model.fit(data, label, epochs=2, batch_size=64, save_dir="mnist_checkpoint")
    eval_out = model.evaluate(data, batch_size=64)
    model.summary()
