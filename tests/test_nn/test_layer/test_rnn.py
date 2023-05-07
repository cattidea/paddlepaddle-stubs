# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from paddle import Tensor
from typing_extensions import assert_type


def test_import():
    paddle.nn.SimpleRNN
    paddle.nn.GRU
    paddle.nn.LSTM

    from paddle.nn import GRU  # pyright: ignore [reportUnusedImport]
    from paddle.nn import LSTM  # pyright: ignore [reportUnusedImport]
    from paddle.nn import SimpleRNN  # pyright: ignore [reportUnusedImport]


def test_types():
    rnn = paddle.nn.SimpleRNN(16, 32, 2)
    x = paddle.randn((4, 23, 16))
    prev_h = paddle.randn((2, 4, 32))
    y, h = rnn(x, prev_h)

    assert_type(rnn, paddle.nn.SimpleRNN)
    assert_type(y, Tensor)
    assert_type(h, Tensor)

    gru = paddle.nn.GRU(16, 32, 2)
    x = paddle.randn((4, 23, 16))
    prev_h = paddle.randn((2, 4, 32))
    y, h = gru(x, prev_h)

    assert_type(gru, paddle.nn.GRU)
    assert_type(y, Tensor)
    assert_type(h, Tensor)

    lstm = paddle.nn.LSTM(16, 32, 2)
    x = paddle.randn((4, 23, 16))
    prev_h = paddle.randn((2, 4, 32))
    y, h = lstm(x, prev_h)

    assert_type(lstm, paddle.nn.LSTM)
    assert_type(y, Tensor)
    assert_type(h, Tensor)
