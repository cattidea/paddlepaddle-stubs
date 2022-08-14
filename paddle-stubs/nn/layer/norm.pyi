from __future__ import annotations

from typing import Any, Optional

from typing_extensions import Literal

from ..._typing import DataLayout1D, DataLayout2D, DataLayout3D, DataLayoutND, Tensor
from ..._typing.basic import IntSequence
from ...fluid.dygraph import BatchNorm as BatchNorm
from ...fluid.dygraph import SpectralNorm as SpectralNorm
from ...framework import ParamAttr
from .. import Layer

class _InstanceNormBase(Layer):
    scale: Any = ...
    bias: Any = ...
    def __init__(
        self,
        num_features: int,
        epsilon: float = ...,
        momentum: float = ...,
        weight_attr: Optional[ParamAttr] = ...,
        bias_attr: Optional[ParamAttr | bool] = ...,
        data_format: DataLayoutND = ...,
        name: Optional[str] = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    __call__ = forward

class InstanceNorm1D(_InstanceNormBase):
    def __init__(
        self,
        num_features: int,
        epsilon: float = ...,
        momentum: float = ...,
        weight_attr: Optional[ParamAttr] = ...,
        bias_attr: Optional[ParamAttr | bool] = ...,
        data_format: DataLayout1D = ...,
        name: Optional[str] = ...,
    ) -> None: ...

class InstanceNorm2D(_InstanceNormBase):
    def __init__(
        self,
        num_features: int,
        epsilon: float = ...,
        momentum: float = ...,
        weight_attr: Optional[ParamAttr] = ...,
        bias_attr: Optional[ParamAttr | bool] = ...,
        data_format: DataLayout2D = ...,
        name: Optional[str] = ...,
    ) -> None: ...

class InstanceNorm3D(_InstanceNormBase):
    def __init__(
        self,
        num_features: int,
        epsilon: float = ...,
        momentum: float = ...,
        weight_attr: Optional[ParamAttr] = ...,
        bias_attr: Optional[ParamAttr | bool] = ...,
        data_format: DataLayout3D = ...,
        name: Optional[str] = ...,
    ) -> None: ...

class GroupNorm(Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        epsilon: float = ...,
        weight_attr: Optional[ParamAttr | bool] = ...,
        bias_attr: Optional[ParamAttr | bool] = ...,
        data_format: Literal["NCHW"] = ...,
        name: Optional[str] = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    __call__ = forward

class LayerNorm(Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        normalized_shape: int | IntSequence,
        epsilon: float = ...,
        weight_attr: Optional[ParamAttr | bool] = ...,
        bias_attr: Optional[ParamAttr | bool] = ...,
        name: Optional[str] = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    __call__ = forward

class _BatchNormBase(Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        num_features: int,
        momentum: float = ...,
        epsilon: float = ...,
        weight_attr: Optional[ParamAttr | bool] = ...,
        bias_attr: Optional[ParamAttr | bool] = ...,
        data_format: DataLayoutND = ...,
        use_global_stats: Optional[bool] = ...,
        name: Optional[str] = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    __call__ = forward

class BatchNorm1D(_BatchNormBase):
    def __init__(
        self,
        num_features: int,
        momentum: float = ...,
        epsilon: float = ...,
        weight_attr: Optional[ParamAttr | bool] = ...,
        bias_attr: Optional[ParamAttr | bool] = ...,
        data_format: DataLayout1D = ...,
        use_global_stats: Optional[bool] = ...,
        name: Optional[str] = ...,
    ) -> None: ...

class BatchNorm2D(_BatchNormBase):
    def __init__(
        self,
        num_features: int,
        momentum: float = ...,
        epsilon: float = ...,
        weight_attr: Optional[ParamAttr | bool] = ...,
        bias_attr: Optional[ParamAttr | bool] = ...,
        data_format: DataLayout2D = ...,
        use_global_stats: Optional[bool] = ...,
        name: Optional[str] = ...,
    ) -> None: ...

class BatchNorm3D(_BatchNormBase):
    def __init__(
        self,
        num_features: int,
        momentum: float = ...,
        epsilon: float = ...,
        weight_attr: Optional[ParamAttr | bool] = ...,
        bias_attr: Optional[ParamAttr | bool] = ...,
        data_format: DataLayout3D = ...,
        use_global_stats: Optional[bool] = ...,
        name: Optional[str] = ...,
    ) -> None: ...

class SyncBatchNorm(_BatchNormBase):
    def __init__(
        self,
        num_features: int,
        momentum: float = ...,
        epsilon: float = ...,
        weight_attr: Optional[ParamAttr | bool] = ...,
        bias_attr: Optional[ParamAttr | bool] = ...,
        data_format: DataLayoutND = ...,
        name: Optional[str] = ...,
    ) -> None: ...
    @classmethod
    def convert_sync_batchnorm(cls, layer: _BatchNormBase) -> SyncBatchNorm: ...

class LocalResponseNorm(Layer):
    size: Any = ...
    alpha: Any = ...
    beta: Any = ...
    k: Any = ...
    data_format: Any = ...
    name: Any = ...
    def __init__(
        self,
        size: int,
        alpha: float = ...,
        beta: float = ...,
        k: float = ...,
        data_format: DataLayoutND = ...,
        name: Optional[str] = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    __call__ = forward
