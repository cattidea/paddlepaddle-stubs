from __future__ import annotations

from typing import Any, Optional

from ...fluid import dygraph_utils as dygraph_utils
from ...fluid.data_feeder import check_type as check_type
from ...fluid.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ...fluid.dygraph import BatchNorm as BatchNorm
from ...fluid.dygraph import SpectralNorm as SpectralNorm
from ...framework import ParamAttr as ParamAttr
from ...framework import get_default_dtype as get_default_dtype
from ...framework import no_grad as no_grad
from ...framework import set_default_dtype as set_default_dtype
from .. import Layer as Layer
from ..functional import batch_norm as batch_norm
from ..functional import instance_norm as instance_norm
from ..functional import layer_norm as layer_norm
from ..initializer import Constant as Constant

class _InstanceNormBase(Layer):
    scale: Any = ...
    bias: Any = ...
    def __init__(
        self,
        num_features: Any,
        epsilon: float = ...,
        momentum: float = ...,
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        data_format: str = ...,
        name: Optional[Any] = ...,
    ) -> None: ...
    def forward(self, input: Any): ...
    def extra_repr(self): ...

class InstanceNorm1D(_InstanceNormBase): ...
class InstanceNorm2D(_InstanceNormBase): ...
class InstanceNorm3D(_InstanceNormBase): ...

class GroupNorm(Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        num_groups: Any,
        num_channels: Any,
        epsilon: float = ...,
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        data_format: str = ...,
        name: Optional[Any] = ...,
    ) -> None: ...
    def forward(self, input: Any): ...
    def extra_repr(self): ...

class LayerNorm(Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        normalized_shape: Any,
        epsilon: float = ...,
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        name: Optional[Any] = ...,
    ) -> None: ...
    def forward(self, input: Any): ...
    def extra_repr(self): ...

class _BatchNormBase(Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        num_features: Any,
        momentum: float = ...,
        epsilon: float = ...,
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        data_format: str = ...,
        use_global_stats: Optional[Any] = ...,
        name: Optional[Any] = ...,
    ) -> None: ...
    def forward(self, input: Any): ...
    def extra_repr(self): ...

class BatchNorm1D(_BatchNormBase):
    def __init__(
        self,
        num_features: Any,
        momentum: float = ...,
        epsilon: float = ...,
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        data_format: str = ...,
        use_global_stats: Optional[Any] = ...,
        name: Optional[Any] = ...,
    ) -> None: ...

class BatchNorm2D(_BatchNormBase): ...

class BatchNorm3D(_BatchNormBase):
    def __init__(
        self,
        num_features: Any,
        momentum: float = ...,
        epsilon: float = ...,
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        data_format: str = ...,
        use_global_stats: Optional[Any] = ...,
        name: Optional[Any] = ...,
    ) -> None: ...

class SyncBatchNorm(_BatchNormBase):
    def __init__(
        self,
        num_features: Any,
        momentum: float = ...,
        epsilon: float = ...,
        weight_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        data_format: str = ...,
        name: Optional[Any] = ...,
    ) -> None: ...
    def forward(self, x: Any): ...
    @classmethod
    def convert_sync_batchnorm(cls, layer: Any): ...

class LocalResponseNorm(Layer):
    size: Any = ...
    alpha: Any = ...
    beta: Any = ...
    k: Any = ...
    data_format: Any = ...
    name: Any = ...
    def __init__(
        self,
        size: Any,
        alpha: float = ...,
        beta: float = ...,
        k: float = ...,
        data_format: str = ...,
        name: Optional[Any] = ...,
    ) -> None: ...
    def forward(self, input: Any): ...
    def extra_repr(self): ...
