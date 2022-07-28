# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from typing_extensions import assert_type


def test_import():
    paddle.regularizer.L1Decay
    paddle.regularizer.L2Decay

    from paddle.regularizer import L1Decay  # pyright: ignore [reportUnusedImport]
    from paddle.regularizer import L2Decay  # pyright: ignore [reportUnusedImport]


def test_types():
    l1 = paddle.regularizer.L1Decay(0.1)
    l2 = paddle.regularizer.L2Decay(0.1)
    assert_type(l1, paddle.regularizer.L1Decay)
    assert_type(l2, paddle.regularizer.L2Decay)
