# pyright: strict, reportUnusedVariable=false

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import paddle
from paddle import Tensor
from typing_extensions import assert_type


def test_import():
    paddle.linalg.inv
    paddle.linalg.cholesky
    paddle.linalg.cholesky_solve
    paddle.linalg.cond
    paddle.linalg.cov
    paddle.linalg.det
    paddle.linalg.eig
    paddle.linalg.eigh
    paddle.linalg.eigvals
    paddle.linalg.eigvalsh
    paddle.linalg.lstsq
    paddle.linalg.lu
    paddle.linalg.lu_unpack
    paddle.linalg.matrix_power
    paddle.linalg.matrix_rank
    paddle.linalg.multi_dot
    paddle.linalg.norm
    paddle.linalg.pinv
    paddle.linalg.qr
    paddle.linalg.slogdet
    paddle.linalg.solve
    paddle.linalg.svd
    paddle.linalg.triangular_solve

    from paddle.linalg import cholesky  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import cholesky_solve  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import cond  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import cov  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import det  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import eig  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import eigh  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import eigvals  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import eigvalsh  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import inv  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import lstsq  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import lu  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import lu_unpack  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import matrix_power  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import matrix_rank  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import multi_dot  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import norm  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import pinv  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import qr  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import slogdet  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import solve  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import svd  # pyright: ignore [reportUnusedImport]
    from paddle.linalg import triangular_solve  # pyright: ignore [reportUnusedImport]


def test_inv():
    mat = paddle.to_tensor([[2, 0], [0, 2]], dtype="float32")
    out = paddle.linalg.inv(mat)
    assert_type(out, Tensor)


def test_cholesky():
    a = np.random.rand(3, 3)
    a_t: npt.NDArray[np.float64] = np.transpose(a, [1, 0])  # type: ignore
    x_data: npt.NDArray[np.float64] = np.matmul(a, a_t) + 1e-03
    x = paddle.to_tensor(x_data)
    out = paddle.linalg.cholesky(x, upper=False)
    assert_type(out, Tensor)


def test_cholesky_solve():
    u = paddle.to_tensor([[1, 1, 1], [0, 2, 1], [0, 0, -1]], dtype="float64")
    b = paddle.to_tensor([[0], [-9], [5]], dtype="float64")
    out = paddle.linalg.cholesky_solve(b, u, upper=True)
    assert_type(out, Tensor)


def test_cond():
    x = paddle.to_tensor([[1.0, 0, -1], [0, 1, 0], [1, 0, 1]])

    paddle.linalg.cond(x, p=-1)
    paddle.linalg.cond(x, p=1)
    paddle.linalg.cond(x, p=-2)
    paddle.linalg.cond(x, p=2)
    paddle.linalg.cond(x, p=float("inf"))
    paddle.linalg.cond(x, p=float("-inf"))
    paddle.linalg.cond(x, p=np.inf)
    paddle.linalg.cond(x, p=-np.inf)
    paddle.linalg.cond(x, p="fro")
    paddle.linalg.cond(x, p="nuc")


def test_cov():
    xt = paddle.rand((3, 4))
    out = paddle.linalg.cov(xt)
    assert_type(out, Tensor)


def test_det():
    x = paddle.randn([3, 3, 3])
    A = paddle.linalg.det(x)
    assert_type(A, Tensor)


def test_eig():
    x_data = paddle.to_tensor(
        [[1.6707249, 7.2249975, 6.5045543], [9.956216, 8.749598, 6.066444], [4.4251957, 1.7983172, 0.370647]],
        dtype="float32",
    )

    w, v = paddle.linalg.eig(x_data)
    assert_type(w, Tensor)
    assert_type(v, Tensor)


def test_eigh():
    x = paddle.to_tensor([[1, -2j], [2j, 5]])
    out_value, out_vector = paddle.linalg.eigh(x)
    assert_type(out_value, Tensor)
    assert_type(out_vector, Tensor)


def test_eigvals():
    x = paddle.rand(shape=[3, 3], dtype="float64")
    out_value = paddle.linalg.eigvals(x)
    assert_type(out_value, Tensor)


def test_eigvalsh():
    x = paddle.to_tensor([[1, -2j], [2j, 5]])
    out_value = paddle.linalg.eigvalsh(x)
    assert_type(out_value, Tensor)


def test_lstsq():
    x = paddle.to_tensor([[1, 3], [3, 2], [5, 6.0]])
    y = paddle.to_tensor([[3, 4, 6], [5, 3, 4], [1, 2, 1.0]])
    solution, residuals, rank, singular_values = paddle.linalg.lstsq(x, y, driver="gelsd")
    assert_type(solution, Tensor)
    assert_type(residuals, Tensor)
    assert_type(rank, Tensor)
    assert_type(singular_values, Tensor)


def test_lu():
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype("float64")
    lu, p, info = paddle.linalg.lu(x, get_infos=True)
    assert_type(lu, Tensor)
    assert_type(p, Tensor)
    assert_type(info, Tensor)


def test_lu_unpack():
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype("float64")
    lu, p, info = paddle.linalg.lu(x, get_infos=True)
    assert_type(lu, Tensor)
    assert_type(p, Tensor)
    assert_type(info, Tensor)


def test_matrix_power():
    x = paddle.to_tensor(
        [
            [1, 2, 3],
            [1, 4, 9],
            [1, 8, 27],
        ],
        dtype="float64",
    )
    out = paddle.linalg.matrix_power(x, 2)
    assert_type(out, Tensor)


def test_matrix_rank():
    c = paddle.ones(shape=[3, 4, 5, 5])
    d = paddle.linalg.matrix_rank(c, tol=0.01, hermitian=True)
    assert_type(d, Tensor)


def test_multi_dot():
    A = paddle.rand([3, 4]).astype(np.float32)
    B = paddle.rand([4, 5]).astype(np.float32)
    out = paddle.linalg.multi_dot([A, B])
    assert_type(out, Tensor)


def test_norm():
    np_input: npt.NDArray[np.float32] = np.arange(24).astype("float32") - 12  # type: ignore
    np_input = np_input.reshape([2, 3, 4])  # type: ignore
    x = paddle.to_tensor(np_input)

    out_fro = paddle.linalg.norm(x, p="fro", axis=[0, 1])
    assert_type(out_fro, Tensor)
    out_pnorm = paddle.linalg.norm(x, p=2, axis=-1)
    assert_type(out_pnorm, Tensor)
    out_pnorm = paddle.linalg.norm(x, p=2, axis=[0, 1])
    assert_type(out_pnorm, Tensor)
    out_pnorm = paddle.linalg.norm(x, p=np.inf, axis=0)
    assert_type(out_pnorm, Tensor)


def test_pinv():
    x: npt.NDArray[np.float64] = paddle.arange(15).reshape((3, 5)).astype("float64")  # type: ignore
    input = paddle.to_tensor(x)
    out = paddle.linalg.pinv(input)
    assert_type(out, Tensor)


def test_qr():
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype("float64")
    q, r = paddle.linalg.qr(x)
    assert_type(q, Tensor)
    assert_type(r, Tensor)


def test_slogdet():
    x = paddle.randn([3, 3, 3])
    A = paddle.linalg.slogdet(x)
    assert_type(A, Tensor)


def test_solve():
    x = paddle.to_tensor([[3, 1], [1, 2]], dtype="float64")
    y = paddle.to_tensor([9, 8], dtype="float64")
    out = paddle.linalg.solve(x, y)
    assert_type(out, Tensor)


def test_svd():
    x = paddle.to_tensor([[1.0, 2.0], [1.0, 3.0], [4.0, 6.0]]).astype("float64")
    x = x.reshape([3, 2])
    u, s, vh = paddle.linalg.svd(x)
    assert_type(u, Tensor)
    assert_type(s, Tensor)
    assert_type(vh, Tensor)


def test_triangular_solve():
    x = paddle.to_tensor(
        [
            [1, 1, 1],
            [0, 2, 1],
            [0, 0, -1],
        ],
        dtype="float64",
    )
    y = paddle.to_tensor([[0], [-9], [5]], dtype="float64")
    out = paddle.linalg.triangular_solve(x, y, upper=True)
    assert_type(out, Tensor)
