from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ...fluid.initializer import NumpyArrayInitializer

class Assign(NumpyArrayInitializer):
    def __init__(
        self,
        value: np.ndarray[Any, Any],
        name: Optional[str] = ...,
    ) -> None: ...
