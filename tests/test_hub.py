# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from typing_extensions import assert_type


def test_import():
    paddle.hub.help
    paddle.hub.list
    paddle.hub.load

    from paddle.hub import help  # pyright: ignore [reportUnusedImport]
    from paddle.hub import list  # pyright: ignore [reportUnusedImport]
    from paddle.hub import load  # pyright: ignore [reportUnusedImport]


def test_help():
    docs = paddle.hub.help("PaddlePaddle/PaddleClas:develop", "alexnet", source="github", force_reload=True)
    assert_type(docs, str)


def test_list():
    models = paddle.hub.list("PaddlePaddle/PaddleClas:develop", source="github", force_reload=True)


def test_load():
    model = paddle.hub.load("PaddlePaddle/PaddleClas:develop", "alexnet", source="github", force_reload=True)
    assert_type(model, paddle.nn.Layer)
