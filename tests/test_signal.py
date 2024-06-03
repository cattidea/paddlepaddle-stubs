# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing_extensions import assert_type

import paddle
from paddle import Tensor


def test_import():
    paddle.signal.stft
    paddle.signal.istft

    from paddle.signal import istft, stft  # pyright: ignore [reportUnusedImport]


def test_types():
    x = paddle.randn([8, 48000], dtype=paddle.float64)
    y1 = paddle.signal.stft(x, n_fft=512)
    y2 = paddle.signal.istft(x, n_fft=512, onesided=False)
    assert_type(y1, Tensor)
    assert_type(y2, Tensor)
