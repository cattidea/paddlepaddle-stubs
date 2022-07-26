from __future__ import annotations

from typing import Any, Optional

from ...fluid.data_feeder import check_type as check_type
from ...fluid.initializer import NumpyArrayInitializer as NumpyArrayInitializer

class Assign(NumpyArrayInitializer):
    def __init__(self, value: Any, name: Optional[Any] = ...) -> None: ...
