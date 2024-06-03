# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing_extensions import assert_type

import paddle
from paddle import Tensor


def test_import():
    paddle.nn.BCEWithLogitsLoss
    paddle.nn.CrossEntropyLoss
    paddle.nn.HSigmoidLoss
    paddle.nn.MSELoss
    paddle.nn.L1Loss
    paddle.nn.BCELoss
    paddle.nn.NLLLoss
    paddle.nn.KLDivLoss
    paddle.nn.MarginRankingLoss
    paddle.nn.CTCLoss
    paddle.nn.SmoothL1Loss
    paddle.nn.HingeEmbeddingLoss

    from paddle.nn import (
        BCELoss,  # pyright: ignore [reportUnusedImport]
        BCEWithLogitsLoss,  # pyright: ignore [reportUnusedImport]
        CrossEntropyLoss,  # pyright: ignore [reportUnusedImport]
        CTCLoss,  # pyright: ignore [reportUnusedImport]
        HingeEmbeddingLoss,  # pyright: ignore [reportUnusedImport]
        HSigmoidLoss,  # pyright: ignore [reportUnusedImport]
        KLDivLoss,  # pyright: ignore [reportUnusedImport]
        L1Loss,  # pyright: ignore [reportUnusedImport]
        MarginRankingLoss,  # pyright: ignore [reportUnusedImport]
        MSELoss,  # pyright: ignore [reportUnusedImport]
        NLLLoss,  # pyright: ignore [reportUnusedImport]
        SmoothL1Loss,  # pyright: ignore [reportUnusedImport]
    )


def test_types():
    logit = paddle.to_tensor([5.0, 1.0, 3.0], dtype="float32")
    label = paddle.to_tensor([1.0, 0.0, 1.0], dtype="float32")

    bce_logit_loss = paddle.nn.BCEWithLogitsLoss()
    output = bce_logit_loss(logit, label)
    assert_type(bce_logit_loss, paddle.nn.BCEWithLogitsLoss)
    assert_type(output, Tensor)

    cross_entropy_loss = paddle.nn.CrossEntropyLoss()
    output = cross_entropy_loss(logit, label)
    assert_type(cross_entropy_loss, paddle.nn.CrossEntropyLoss)
    assert_type(output, Tensor)

    hsigmoid_loss = paddle.nn.HSigmoidLoss(3, 5)
    output = hsigmoid_loss(logit, label)
    assert_type(hsigmoid_loss, paddle.nn.HSigmoidLoss)
    assert_type(output, Tensor)

    mse_loss = paddle.nn.MSELoss()
    output = mse_loss(logit, label)
    assert_type(mse_loss, paddle.nn.MSELoss)
    assert_type(output, Tensor)

    l1_loss = paddle.nn.L1Loss()
    output = l1_loss(logit, label)
    assert_type(l1_loss, paddle.nn.L1Loss)
    assert_type(output, Tensor)

    bce_loss = paddle.nn.BCELoss()
    output = bce_loss(logit, label)
    assert_type(bce_loss, paddle.nn.BCELoss)
    assert_type(output, Tensor)

    nll_loss = paddle.nn.NLLLoss()
    output = nll_loss(logit, label)
    assert_type(nll_loss, paddle.nn.NLLLoss)
    assert_type(output, Tensor)

    kldiv_loss = paddle.nn.KLDivLoss()
    output = kldiv_loss(logit, label)
    assert_type(kldiv_loss, paddle.nn.KLDivLoss)
    assert_type(output, Tensor)

    margin_ranking_loss = paddle.nn.MarginRankingLoss()
    output = margin_ranking_loss(logit, logit, label)
    assert_type(margin_ranking_loss, paddle.nn.MarginRankingLoss)
    assert_type(output, Tensor)

    ctc_loss = paddle.nn.CTCLoss()
    output = ctc_loss(logit, label, paddle.to_tensor([1]), paddle.to_tensor([1]))
    assert_type(ctc_loss, paddle.nn.CTCLoss)
    assert_type(output, Tensor)

    smooth_l1_loss = paddle.nn.SmoothL1Loss()
    output = smooth_l1_loss(logit, label)
    assert_type(smooth_l1_loss, paddle.nn.SmoothL1Loss)
    assert_type(output, Tensor)

    hinge_embedding_loss = paddle.nn.HingeEmbeddingLoss()
    output = hinge_embedding_loss(logit, label)
    assert_type(hinge_embedding_loss, paddle.nn.HingeEmbeddingLoss)
    assert_type(output, Tensor)
