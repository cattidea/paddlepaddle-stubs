# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle
from typing_extensions import assert_type


def test_import():
    paddle.version.cuda
    paddle.version.cudnn
    paddle.version.show

    from paddle.version import cuda  # pyright: ignore [reportUnusedImport]
    from paddle.version import cudnn  # pyright: ignore [reportUnusedImport]
    from paddle.version import show  # pyright: ignore [reportUnusedImport]


def test_types():
    cuda_ver: str = paddle.version.cuda()
    cudnn_ver: str = paddle.version.cudnn()
    paddle.version.show()
    assert_type(cuda_ver, str)
    assert_type(cudnn_ver, str)
