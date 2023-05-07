# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from paddle import Tensor
from typing_extensions import assert_type


def test_import():
    paddle.nn.Transformer

    from paddle.nn import Transformer  # pyright: ignore [reportUnusedImport]


def test_types():
    enc_input = paddle.rand((2, 4, 128))
    dec_input = paddle.rand((2, 6, 128))
    enc_self_attn_mask = paddle.rand((2, 2, 4, 4))
    dec_self_attn_mask = paddle.rand((2, 2, 6, 6))
    cross_attn_mask = paddle.rand((2, 2, 6, 4))
    transformer = paddle.nn.Transformer(128, 2, 4, 4, 512)
    output = transformer(
        enc_input,
        dec_input,
        enc_self_attn_mask,
        dec_self_attn_mask,
        cross_attn_mask,
    )
    assert_type(transformer, paddle.nn.Transformer)
    assert_type(output, Tensor)
