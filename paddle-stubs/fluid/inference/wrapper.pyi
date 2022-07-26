from __future__ import annotations

from typing import Any

from .. import core as core
from ..core import AnalysisConfig as AnalysisConfig
from ..core import PaddleDType as PaddleDType
from ..core import PaddleInferPredictor as PaddleInferPredictor
from ..core import PaddleInferTensor as PaddleInferTensor
from ..core import PaddlePlace as PaddlePlace

DataType = PaddleDType
PlaceType = PaddlePlace
PrecisionType: Any
Config = AnalysisConfig
Tensor = PaddleInferTensor
Predictor = PaddleInferPredictor

def tensor_copy_from_cpu(self, data: Any) -> None: ...
def tensor_share_external_data(self, data: Any) -> None: ...
