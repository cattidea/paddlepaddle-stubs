# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from paddle import Tensor


def test_full_path_access():
    x = paddle.randn([8, 48000], dtype=paddle.float64)
    y1 = paddle.signal.stft(x, n_fft=512)
    y2 = paddle.signal.istft(x, n_fft=512, onesided=False)
    _ = y1 - y2


def test_full_path_import():
    from paddle.signal import istft, stft

    x = paddle.randn([8, 48000], dtype=paddle.float64)
    y1 = stft(x, n_fft=512)
    y2 = istft(x, n_fft=512, onesided=False)
    _ = y1 - y2


def test_types():
    x = paddle.randn([8, 48000], dtype=paddle.float64)
    y1 = paddle.signal.stft(x, n_fft=512)
    y2 = paddle.signal.istft(x, n_fft=512, onesided=False)
    assert isinstance(y1, Tensor)
    assert isinstance(y2, Tensor)
