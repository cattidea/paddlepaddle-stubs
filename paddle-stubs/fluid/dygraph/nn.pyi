from __future__ import annotations

from typing import Any, Optional

from ..._typing import DataLayoutND, DTypeLike, ShapeLike, Tensor
from ..param_attr import ParamAttr
from . import layers

class Conv2D(layers.Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        num_channels: Any,
        num_filters: Any,
        filter_size: Any,
        stride: int = ...,
        padding: int = ...,
        dilation: int = ...,
        groups: Any | None = ...,
        param_attr: Any | None = ...,
        bias_attr: Any | None = ...,
        use_cudnn: bool = ...,
        act: Any | None = ...,
        dtype: str = ...,
    ): ...
    def forward(self, input: Any): ...

class Conv3D(layers.Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        num_channels: Any,
        num_filters: Any,
        filter_size: Any,
        stride: int = ...,
        padding: int = ...,
        dilation: int = ...,
        groups: Any | None = ...,
        param_attr: Any | None = ...,
        bias_attr: Any | None = ...,
        use_cudnn: bool = ...,
        act: Any | None = ...,
        dtype: str = ...,
    ): ...
    def forward(self, input: Any): ...

class Conv3DTranspose(layers.Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        num_channels: Any,
        num_filters: Any,
        filter_size: Any,
        padding: int = ...,
        stride: int = ...,
        dilation: int = ...,
        groups: Any | None = ...,
        param_attr: Any | None = ...,
        bias_attr: Any | None = ...,
        use_cudnn: bool = ...,
        act: Any | None = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class Pool2D(layers.Layer):
    def __init__(
        self,
        pool_size: int = ...,
        pool_type: str = ...,
        pool_stride: int = ...,
        pool_padding: int = ...,
        global_pooling: bool = ...,
        use_cudnn: bool = ...,
        ceil_mode: bool = ...,
        exclusive: bool = ...,
        data_format: str = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class Linear(layers.Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        input_dim: Any,
        output_dim: Any,
        param_attr: Any | None = ...,
        bias_attr: Any | None = ...,
        act: Any | None = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class InstanceNorm(layers.Layer):
    scale: Any = ...
    bias: Any = ...
    def __init__(
        self,
        num_channels: Any,
        epsilon: float = ...,
        param_attr: Any | None = ...,
        bias_attr: Any | None = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class BatchNorm(layers.Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        num_channels: int,
        act: str | None = ...,
        is_test: bool = ...,
        momentum: float = ...,
        epsilon: float = ...,
        param_attr: ParamAttr | None = ...,
        bias_attr: ParamAttr | bool | None = ...,
        dtype: DTypeLike = ...,
        data_layout: DataLayoutND = ...,
        in_place: bool = ...,
        moving_mean_name: str | None = ...,
        moving_variance_name: str | None = ...,
        do_model_average_for_mean_and_var: bool = ...,
        use_global_stats: bool = ...,
        trainable_statistics: bool = ...,
    ) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    __call__ = forward

class Dropout(layers.Layer):
    def __init__(
        self, p: float = ..., seed: Any | None = ..., dropout_implementation: str = ..., is_test: bool = ...
    ) -> None: ...
    def forward(self, input: Any): ...

class Embedding(layers.Layer):
    weight: Any = ...
    def __init__(
        self,
        size: Any,
        is_sparse: bool = ...,
        is_distributed: bool = ...,
        padding_idx: Any | None = ...,
        param_attr: Any | None = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class LayerNorm(layers.Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        normalized_shape: Any,
        scale: bool = ...,
        shift: bool = ...,
        epsilon: float = ...,
        param_attr: Any | None = ...,
        bias_attr: Any | None = ...,
        act: Any | None = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class GRUUnit(layers.Layer):
    activation: Any = ...
    gate_activation: Any = ...
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        size: Any,
        param_attr: Any | None = ...,
        bias_attr: Any | None = ...,
        activation: str = ...,
        gate_activation: str = ...,
        origin_mode: bool = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, input: Any, hidden: Any): ...

class NCE(layers.Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        num_total_classes: Any,
        dim: Any,
        sample_weight: Any | None = ...,
        param_attr: Any | None = ...,
        bias_attr: Any | None = ...,
        num_neg_samples: Any | None = ...,
        sampler: str = ...,
        custom_dist: Any | None = ...,
        seed: int = ...,
        is_sparse: bool = ...,
        dtype: str = ...,
    ): ...
    def forward(self, input: Any, label: Any, sample_weight: Any | None = ...): ...

class PRelu(layers.Layer):
    weight: Any = ...
    def __init__(
        self,
        mode: Any,
        channel: Any | None = ...,
        input_shape: Any | None = ...,
        param_attr: Any | None = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class BilinearTensorProduct(layers.Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        input1_dim: Any,
        input2_dim: Any,
        output_dim: Any,
        name: str | None = ...,
        act: Any | None = ...,
        param_attr: Any | None = ...,
        bias_attr: Any | None = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, x: Any, y: Any): ...

class Conv2DTranspose(layers.Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        num_channels: Any,
        num_filters: Any,
        filter_size: Any,
        output_size: Any | None = ...,
        padding: int = ...,
        stride: int = ...,
        dilation: int = ...,
        groups: Any | None = ...,
        param_attr: Any | None = ...,
        bias_attr: Any | None = ...,
        use_cudnn: bool = ...,
        act: Any | None = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class SequenceConv(layers.Layer):
    def __init__(
        self,
        name_scope: Any,
        num_filters: Any,
        filter_size: int = ...,
        filter_stride: int = ...,
        padding: Any | None = ...,
        bias_attr: Any | None = ...,
        param_attr: Any | None = ...,
        act: Any | None = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class RowConv(layers.Layer):
    def __init__(
        self, name_scope: Any, future_context_size: Any, param_attr: Any | None = ..., act: Any | None = ...
    ) -> None: ...
    def forward(self, input: Any): ...

class GroupNorm(layers.Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        channels: Any,
        groups: Any,
        epsilon: float = ...,
        param_attr: Any | None = ...,
        bias_attr: Any | None = ...,
        act: Any | None = ...,
        data_layout: str = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class SpectralNorm(layers.Layer):
    weight_u: Any = ...
    weight_v: Any = ...
    def __init__(
        self,
        weight_shape: ShapeLike,
        dim: int = ...,
        power_iters: int = ...,
        eps: float = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, weight: Tensor) -> Tensor: ...
    __call__ = forward

class TreeConv(layers.Layer):
    bias: Any = ...
    weight: Any = ...
    def __init__(
        self,
        feature_size: Any,
        output_size: Any,
        num_filters: int = ...,
        max_depth: int = ...,
        act: str = ...,
        param_attr: Any | None = ...,
        bias_attr: Any | None = ...,
        name: str | None = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, nodes_vector: Any, edge_set: Any): ...

class Flatten(layers.Layer):
    start_axis: Any = ...
    stop_axis: Any = ...
    def __init__(self, start_axis: int = ..., stop_axis: int = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    __call__ = forward
