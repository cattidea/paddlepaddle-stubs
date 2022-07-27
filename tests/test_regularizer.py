# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle


def test_full_path_access():
    l1 = paddle.regularizer.L1Decay(0.1)
    l2 = paddle.regularizer.L2Decay(0.1)


def test_full_path_import():
    from paddle.regularizer import L1Decay, L2Decay

    l1 = L1Decay(0.1)
    l2 = L2Decay(0.1)
