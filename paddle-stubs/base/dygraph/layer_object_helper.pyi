from __future__ import annotations

from typing import Any, Optional

from .. import core as core
from ..framework import Parameter as Parameter
from ..layer_helper_base import LayerHelperBase as LayerHelperBase
from ..param_attr import ParamAttr as ParamAttr

class LayerObjectHelper(LayerHelperBase):
    def __init__(self, name: Any) -> None: ...
    def append_op(
        self,
        type: Any | None = ...,
        inputs: Any | None = ...,
        outputs: Any | None = ...,
        attrs: Any | None = ...,
        stop_gradient: Any | None = ...,
    ): ...
    def iter_inputs_and_params(self, inputs_in: Any, param_attr_in: Any | None = ...) -> None: ...
    def input_dtype(self, inputs_in: Any): ...
    def get_parameter(self, name: Any): ...
    def append_activation(self, input_var: Any, act: Any | None = ..., use_cudnn: Any | None = ...): ...
    def is_instance(self, param: Any, cls: Any) -> None: ...
