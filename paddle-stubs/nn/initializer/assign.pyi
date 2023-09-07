from __future__ import annotations

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from ...base.initializer import NumpyArrayInitializer

class Assign(NumpyArrayInitializer):
    def __init__(
        self,
        value: npt.NDArray[Any],
        name: str | None = ...,
    ) -> None: ...
