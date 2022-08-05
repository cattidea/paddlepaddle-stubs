from __future__ import annotations

from typing import Any

from .. import core as core

DataType = Any
PlaceType = Any
PrecisionType: Any
Config = Any
Tensor = Any
Predictor = Any

def tensor_copy_from_cpu(self, data: Any) -> None: ...
def tensor_share_external_data(self, data: Any) -> None: ...
