# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle


def test_import():
    paddle.optimizer.lr.LRScheduler
    paddle.optimizer.lr.NoamDecay
    paddle.optimizer.lr.PiecewiseDecay
    paddle.optimizer.lr.NaturalExpDecay
    paddle.optimizer.lr.InverseTimeDecay
    paddle.optimizer.lr.PolynomialDecay
    paddle.optimizer.lr.LinearWarmup
    paddle.optimizer.lr.ExponentialDecay
    paddle.optimizer.lr.MultiStepDecay
    paddle.optimizer.lr.StepDecay
    paddle.optimizer.lr.LambdaDecay
    paddle.optimizer.lr.ReduceOnPlateau
    paddle.optimizer.lr.CosineAnnealingDecay
    paddle.optimizer.lr.MultiplicativeDecay

    from paddle.optimizer.lr import (
        CosineAnnealingDecay,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.optimizer.lr import (
        ExponentialDecay,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.optimizer.lr import (
        InverseTimeDecay,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.optimizer.lr import LambdaDecay  # pyright: ignore [reportUnusedImport]
    from paddle.optimizer.lr import LinearWarmup  # pyright: ignore [reportUnusedImport]
    from paddle.optimizer.lr import LRScheduler  # pyright: ignore [reportUnusedImport]
    from paddle.optimizer.lr import (
        MultiplicativeDecay,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.optimizer.lr import (
        MultiStepDecay,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.optimizer.lr import (
        NaturalExpDecay,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.optimizer.lr import NoamDecay  # pyright: ignore [reportUnusedImport]
    from paddle.optimizer.lr import (
        PiecewiseDecay,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.optimizer.lr import (
        PolynomialDecay,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.optimizer.lr import (
        ReduceOnPlateau,  # pyright: ignore [reportUnusedImport]
    )
    from paddle.optimizer.lr import StepDecay  # pyright: ignore [reportUnusedImport]


def test_types():
    lr_scheduler = paddle.optimizer.lr.NoamDecay(d_model=10, warmup_steps=100, verbose=True)
    for _ in range(10):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()

    lr_scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[1, 2, 3], values=[0.1, 0.2, 0.3], verbose=True)
    for _ in range(10):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()

    lr_scheduler = paddle.optimizer.lr.NaturalExpDecay(learning_rate=0.1, gamma=0.1, verbose=True)
    for _ in range(10):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()

    lr_scheduler = paddle.optimizer.lr.InverseTimeDecay(learning_rate=0.1, gamma=0.1, verbose=True)
    for _ in range(10):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()

    lr_scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=0.1, decay_steps=10, verbose=True)
    for _ in range(10):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()

    lr_scheduler = paddle.optimizer.lr.LinearWarmup(
        learning_rate=0.1, warmup_steps=10, start_lr=0, end_lr=0.1, verbose=True
    )
    for _ in range(10):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()

    lr_scheduler = paddle.optimizer.lr.ExponentialDecay(learning_rate=0.1, gamma=0.9, verbose=True)
    for _ in range(10):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()

    lr_scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=0.1, milestones=[1, 2, 3], verbose=True)
    for _ in range(10):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()

    lr_scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.5, step_size=5, gamma=0.8, verbose=True)
    for _ in range(10):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()

    lr_scheduler = paddle.optimizer.lr.LambdaDecay(learning_rate=0.5, lr_lambda=lambda x: 0.95**x, verbose=True)
    for _ in range(10):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()

    lr_scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=0.5, verbose=True)
    for _ in range(10):
        lr_scheduler.step(paddle.to_tensor(1.0))
        lr = lr_scheduler.get_lr()

    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=0.5, T_max=10, verbose=True)
    for _ in range(10):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()

    lr_scheduler = paddle.optimizer.lr.MultiplicativeDecay(learning_rate=0.5, lr_lambda=lambda x: 0.95, verbose=True)
    for _ in range(10):
        lr_scheduler.step()
        lr = lr_scheduler.get_lr()
