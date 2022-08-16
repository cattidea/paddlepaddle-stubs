# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import paddle
from typing_extensions import assert_type


def test_import():
    paddle.to_tensor

    from paddle import to_tensor  # pyright: ignore [reportUnusedImport]


def test_to_tensor():
    paddle.to_tensor(1)
    paddle.to_tensor(1.0)
    paddle.to_tensor(1.0 + 1.0j)

    paddle.to_tensor([1])
    paddle.to_tensor([[1, 1], [1, 1]])
    paddle.to_tensor(
        [
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
        ]
    )

    paddle.to_tensor((1,))
    paddle.to_tensor(((1, 1), (1, 1)))
    paddle.to_tensor(
        (
            (
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
            ),
            (
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
            ),
            (
                (1, 1, 1),
                (1, 1, 1),
                (1, 1, 1),
            ),
        )
    )

    data_np: npt.NDArray[Any] = np.array(1.0, dtype=np.float64)  # type: ignore
    paddle.to_tensor(data_np)
    data_np: npt.NDArray[Any] = np.array([1.0], dtype=np.float64)  # type: ignore
    paddle.to_tensor(data_np)

    out = paddle.to_tensor(paddle.to_tensor([1.0]))

    assert_type(out, paddle.Tensor)
