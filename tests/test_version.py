# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing_extensions import assert_type

import paddle


def test_import():
    paddle.version.cuda
    paddle.version.cudnn
    paddle.version.show

    from paddle.version import (
        cuda,  # pyright: ignore [reportUnusedImport]
        cudnn,  # pyright: ignore [reportUnusedImport]
        show,  # pyright: ignore [reportUnusedImport]
    )


def test_types():
    cuda_ver: str = paddle.version.cuda()
    cudnn_ver: str = paddle.version.cudnn()
    paddle.version.show()
    assert_type(cuda_ver, str)
    assert_type(cudnn_ver, str)
