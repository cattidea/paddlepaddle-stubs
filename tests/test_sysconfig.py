# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import paddle


def test_full_path_access():
    paddle.sysconfig.get_lib()
    paddle.sysconfig.get_include()


def test_full_path_import():
    from paddle.sysconfig import get_include, get_lib

    get_lib()
    get_include()


def test_types():
    res_get_lib: str = paddle.sysconfig.get_lib()
    res_get_include: str = paddle.sysconfig.get_include()
    assert isinstance(res_get_lib, str)
    assert isinstance(res_get_include, str)
