# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

from typing_extensions import assert_type

import paddle


def test_import():
    paddle.sysconfig.get_lib
    paddle.sysconfig.get_include

    from paddle.sysconfig import (
        get_include,  # pyright: ignore [reportUnusedImport]
        get_lib,  # pyright: ignore [reportUnusedImport]
    )


def test_types():
    res_get_lib: str = paddle.sysconfig.get_lib()
    res_get_include: str = paddle.sysconfig.get_include()
    assert_type(res_get_lib, str)
    assert_type(res_get_include, str)
