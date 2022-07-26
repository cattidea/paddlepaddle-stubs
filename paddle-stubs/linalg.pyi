from __future__ import annotations

from .tensor import inverse as inv
from .tensor.linalg import cholesky as cholesky
from .tensor.linalg import cholesky_solve as cholesky_solve
from .tensor.linalg import cond as cond
from .tensor.linalg import cov as cov
from .tensor.linalg import det as det
from .tensor.linalg import eig as eig
from .tensor.linalg import eigh as eigh
from .tensor.linalg import eigvals as eigvals
from .tensor.linalg import eigvalsh as eigvalsh
from .tensor.linalg import lstsq as lstsq
from .tensor.linalg import lu as lu
from .tensor.linalg import lu_unpack as lu_unpack
from .tensor.linalg import matrix_power as matrix_power
from .tensor.linalg import matrix_rank as matrix_rank
from .tensor.linalg import multi_dot as multi_dot
from .tensor.linalg import norm as norm
from .tensor.linalg import pinv as pinv
from .tensor.linalg import qr as qr
from .tensor.linalg import slogdet as slogdet
from .tensor.linalg import solve as solve
from .tensor.linalg import svd as svd
from .tensor.linalg import triangular_solve as triangular_solve
