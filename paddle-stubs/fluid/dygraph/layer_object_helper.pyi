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
        type: Optional[Any] = ...,
        inputs: Optional[Any] = ...,
        outputs: Optional[Any] = ...,
        attrs: Optional[Any] = ...,
        stop_gradient: Optional[Any] = ...,
    ): ...
    def iter_inputs_and_params(self, inputs_in: Any, param_attr_in: Optional[Any] = ...) -> None: ...
    def input_dtype(self, inputs_in: Any): ...
    def get_parameter(self, name: Any): ...
    def append_activation(self, input_var: Any, act: Optional[Any] = ..., use_cudnn: Optional[Any] = ...): ...
    def is_instance(self, param: Any, cls: Any) -> None: ...
