# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle


def test_import():
    paddle.optimizer.Adadelta
    paddle.optimizer.Adagrad
    paddle.optimizer.Adam
    paddle.optimizer.Adamax
    paddle.optimizer.AdamW
    paddle.optimizer.Lamb
    paddle.optimizer.Momentum
    paddle.optimizer.Optimizer
    paddle.optimizer.RMSProp
    paddle.optimizer.SGD

    from paddle.optimizer import SGD  # pyright: ignore [reportUnusedImport]
    from paddle.optimizer import Adadelta  # pyright: ignore [reportUnusedImport]
    from paddle.optimizer import Adagrad  # pyright: ignore [reportUnusedImport]
    from paddle.optimizer import Adam  # pyright: ignore [reportUnusedImport]
    from paddle.optimizer import Adamax  # pyright: ignore [reportUnusedImport]
    from paddle.optimizer import AdamW  # pyright: ignore [reportUnusedImport]
    from paddle.optimizer import Lamb  # pyright: ignore [reportUnusedImport]
    from paddle.optimizer import Momentum  # pyright: ignore [reportUnusedImport]
    from paddle.optimizer import Optimizer  # pyright: ignore [reportUnusedImport]
    from paddle.optimizer import RMSProp  # pyright: ignore [reportUnusedImport]


def test_types():
    input = paddle.randn([10, 10]).astype("float32")
    linear = paddle.nn.Linear(10, 10)
    out = linear(input)

    linear_1 = paddle.nn.Linear(10, 10)
    linear_2 = paddle.nn.Linear(10, 10)
    input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
    out = linear_1(input)
    out = linear_2(out)
    loss = paddle.mean(out)

    # Adadelta
    adadelta = paddle.optimizer.Adadelta(
        learning_rate=0.1,
        parameters=[
            {"params": linear_1.parameters()},
            {
                "params": linear_2.parameters(),
                "weight_decay": 0.001,
                "learning_rate": 0.1,
            },
        ],
        weight_decay=0.01,
    )
    loss.backward()
    adadelta.step()
    adadelta.clear_grad()

    # Adagrad
    adagrad = paddle.optimizer.Adagrad(
        learning_rate=0.1,
        parameters=[
            {"params": linear_1.parameters()},
            {
                "params": linear_2.parameters(),
                "weight_decay": 0.001,
                "learning_rate": 0.1,
            },
        ],
        weight_decay=0.01,
    )
    loss.backward()
    adagrad.step()
    adagrad.clear_grad()

    # Adam
    adam = paddle.optimizer.Adam(
        learning_rate=0.1,
        parameters=[
            {"params": linear_1.parameters()},
            {
                "params": linear_2.parameters(),
                "weight_decay": 0.001,
                "learning_rate": 0.1,
                "beta1": 0.8,
            },
        ],
        weight_decay=0.01,
        beta1=0.9,
    )
    loss.backward()
    adam.step()
    adam.clear_grad()

    # Adamax
    adamax = paddle.optimizer.Adamax(
        learning_rate=0.1,
        parameters=[
            {"params": linear_1.parameters()},
            {
                "params": linear_2.parameters(),
                "weight_decay": 0.001,
                "learning_rate": 0.1,
                "beta1": 0.8,
            },
        ],
        weight_decay=0.01,
        beta1=0.9,
    )
    loss.backward()
    adamax.step()
    adamax.clear_grad()

    # AdamW
    adamw = paddle.optimizer.AdamW(
        learning_rate=0.1,
        parameters=[
            {"params": linear_1.parameters()},
            {
                "params": linear_2.parameters(),
                "weight_decay": 0.001,
                "learning_rate": 0.1,
                "beta1": 0.8,
            },
        ],
        weight_decay=0.01,
        beta1=0.9,
    )
    loss.backward()
    adamw.step()
    adamw.clear_grad()

    # Lamb
    lamb = paddle.optimizer.Lamb(
        learning_rate=0.1,
        parameters=[
            {"params": linear_1.parameters()},
            {
                "params": linear_2.parameters(),
                "weight_decay": 0.001,
                "learning_rate": 0.1,
            },
        ],
    )
    loss.backward()
    lamb.step()
    lamb.clear_grad()

    # Momentum
    momentum = paddle.optimizer.Momentum(
        learning_rate=0.1,
        parameters=[
            {"params": linear_1.parameters()},
            {
                "params": linear_2.parameters(),
                "weight_decay": 0.001,
                "learning_rate": 0.1,
            },
        ],
        momentum=0.9,
    )
    loss.backward()
    momentum.step()
    momentum.clear_grad()

    # RMSProp
    rmsprop = paddle.optimizer.RMSProp(
        learning_rate=0.1,
        parameters=[
            {"params": linear_1.parameters()},
            {
                "params": linear_2.parameters(),
                "weight_decay": 0.001,
                "learning_rate": 0.1,
            },
        ],
        weight_decay=0.01,
        momentum=0.9,
    )
    loss.backward()
    rmsprop.step()
    rmsprop.clear_grad()

    # SGD
    sgd = paddle.optimizer.SGD(
        learning_rate=0.1,
        parameters=[
            {"params": linear_1.parameters()},
            {
                "params": linear_2.parameters(),
                "weight_decay": 0.001,
                "learning_rate": 0.1,
            },
        ],
        weight_decay=0.01,
    )
    loss.backward()
    sgd.step()
    sgd.clear_grad()
