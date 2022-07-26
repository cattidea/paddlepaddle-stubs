from __future__ import annotations

from ..fluid.inference import Config as Config
from ..fluid.inference import DataType as DataType
from ..fluid.inference import PlaceType as PlaceType
from ..fluid.inference import PrecisionType as PrecisionType
from ..fluid.inference import Predictor as Predictor
from ..fluid.inference import PredictorPool as PredictorPool
from ..fluid.inference import Tensor as Tensor
from ..fluid.inference import create_predictor as create_predictor
from ..fluid.inference import get_num_bytes_of_data_type as get_num_bytes_of_data_type
from ..fluid.inference import get_trt_compile_version as get_trt_compile_version
from ..fluid.inference import get_trt_runtime_version as get_trt_runtime_version
from ..fluid.inference import get_version as get_version
