from __future__ import annotations

from ...fluid.contrib.mixed_precision import (
    AutoMixedPrecisionLists as AutoMixedPrecisionLists,
)
from ...fluid.contrib.mixed_precision import CustomOpLists as CustomOpLists
from ...fluid.contrib.mixed_precision import bf16 as bf16
from ...fluid.contrib.mixed_precision import cast_model_to_fp16 as cast_model_to_fp16
from ...fluid.contrib.mixed_precision import (
    cast_parameters_to_fp16 as cast_parameters_to_fp16,
)
from ...fluid.contrib.mixed_precision import decorate as decorate
from ...fluid.contrib.mixed_precision import fp16_guard as fp16_guard
