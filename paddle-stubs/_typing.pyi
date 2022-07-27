from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import Self

Numberic = int | float | complex | np.number[Any]

class Tensor:
    def __sub__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
    def __rsub__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
