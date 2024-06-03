# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import numpy as np

import paddle


def test_import():
    paddle.nn.initializer.Bilinear
    paddle.nn.initializer.Initializer
    paddle.nn.initializer.set_global_initializer
    paddle.nn.initializer.Assign
    paddle.nn.initializer.Constant
    paddle.nn.initializer.Dirac
    paddle.nn.initializer.KaimingNormal
    paddle.nn.initializer.KaimingUniform
    paddle.nn.initializer.Normal
    paddle.nn.initializer.TruncatedNormal
    paddle.nn.initializer.Orthogonal
    paddle.nn.initializer.Uniform
    paddle.nn.initializer.XavierNormal
    paddle.nn.initializer.XavierUniform

    from paddle.nn.initializer import (
        Assign,  # pyright: ignore [reportUnusedImport]
        Bilinear,  # pyright: ignore [reportUnusedImport]
        Constant,  # pyright: ignore [reportUnusedImport]
        Dirac,  # pyright: ignore [reportUnusedImport]
        Initializer,  # pyright: ignore [reportUnusedImport]
        KaimingNormal,  # pyright: ignore [reportUnusedImport]
        KaimingUniform,  # pyright: ignore [reportUnusedImport]
        Normal,  # pyright: ignore [reportUnusedImport]
        Orthogonal,  # pyright: ignore [reportUnusedImport]
        TruncatedNormal,  # pyright: ignore [reportUnusedImport]
        Uniform,  # pyright: ignore [reportUnusedImport]
        XavierNormal,  # pyright: ignore [reportUnusedImport]
        XavierUniform,  # pyright: ignore [reportUnusedImport]
        set_global_initializer,  # pyright: ignore [reportUnusedImport]
    )


def test_types():
    tensor = paddle.randint(0, 255, shape=[3, 224, 224])

    init = paddle.nn.initializer.Assign(np.array([1]))  # type: ignore
    init(tensor)

    init = paddle.nn.initializer.Bilinear()
    init(tensor)

    init = paddle.nn.initializer.Constant(0.1)
    init(tensor)

    init = paddle.nn.initializer.Dirac()
    init(tensor)

    init = paddle.nn.initializer.KaimingNormal()
    init(tensor)

    init = paddle.nn.initializer.KaimingUniform()
    init(tensor)

    init = paddle.nn.initializer.Normal(0.1)
    init(tensor)

    init = paddle.nn.initializer.Orthogonal()
    init(tensor)

    init = paddle.nn.initializer.TruncatedNormal(0.1)
    init(tensor)

    init = paddle.nn.initializer.Uniform(0.1)
    init(tensor)

    init = paddle.nn.initializer.XavierNormal()
    init(tensor)

    init = paddle.nn.initializer.XavierUniform()
    init(tensor)

    paddle.nn.initializer.set_global_initializer(
        paddle.nn.initializer.Constant(0.1),
        paddle.nn.initializer.Constant(0.1),
    )
