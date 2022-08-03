from __future__ import annotations

from typing import Any, Optional

from ..._typing import Tensor
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
        groups: Optional[Any] = ...,
        param_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        use_cudnn: bool = ...,
        act: Optional[Any] = ...,
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
        groups: Optional[Any] = ...,
        param_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        use_cudnn: bool = ...,
        act: Optional[Any] = ...,
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
        groups: Optional[Any] = ...,
        param_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        use_cudnn: bool = ...,
        act: Optional[Any] = ...,
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
        param_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        act: Optional[Any] = ...,
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
        param_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class BatchNorm(layers.Layer):
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        num_channels: Any,
        act: Optional[Any] = ...,
        is_test: bool = ...,
        momentum: float = ...,
        epsilon: float = ...,
        param_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        dtype: str = ...,
        data_layout: str = ...,
        in_place: bool = ...,
        moving_mean_name: Optional[Any] = ...,
        moving_variance_name: Optional[Any] = ...,
        do_model_average_for_mean_and_var: bool = ...,
        use_global_stats: bool = ...,
        trainable_statistics: bool = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class Dropout(layers.Layer):
    def __init__(
        self, p: float = ..., seed: Optional[Any] = ..., dropout_implementation: str = ..., is_test: bool = ...
    ) -> None: ...
    def forward(self, input: Any): ...

class Embedding(layers.Layer):
    weight: Any = ...
    def __init__(
        self,
        size: Any,
        is_sparse: bool = ...,
        is_distributed: bool = ...,
        padding_idx: Optional[Any] = ...,
        param_attr: Optional[Any] = ...,
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
        param_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        act: Optional[Any] = ...,
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
        param_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
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
        sample_weight: Optional[Any] = ...,
        param_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        num_neg_samples: Optional[Any] = ...,
        sampler: str = ...,
        custom_dist: Optional[Any] = ...,
        seed: int = ...,
        is_sparse: bool = ...,
        dtype: str = ...,
    ): ...
    def forward(self, input: Any, label: Any, sample_weight: Optional[Any] = ...): ...

class PRelu(layers.Layer):
    weight: Any = ...
    def __init__(
        self,
        mode: Any,
        channel: Optional[Any] = ...,
        input_shape: Optional[Any] = ...,
        param_attr: Optional[Any] = ...,
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
        name: Optional[Any] = ...,
        act: Optional[Any] = ...,
        param_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
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
        output_size: Optional[Any] = ...,
        padding: int = ...,
        stride: int = ...,
        dilation: int = ...,
        groups: Optional[Any] = ...,
        param_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        use_cudnn: bool = ...,
        act: Optional[Any] = ...,
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
        padding: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        param_attr: Optional[Any] = ...,
        act: Optional[Any] = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class RowConv(layers.Layer):
    def __init__(
        self, name_scope: Any, future_context_size: Any, param_attr: Optional[Any] = ..., act: Optional[Any] = ...
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
        param_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        act: Optional[Any] = ...,
        data_layout: str = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, input: Any): ...

class SpectralNorm(layers.Layer):
    weight_u: Any = ...
    weight_v: Any = ...
    def __init__(
        self, weight_shape: Any, dim: int = ..., power_iters: int = ..., eps: float = ..., dtype: str = ...
    ) -> None: ...
    def forward(self, weight: Any): ...

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
        param_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        name: Optional[Any] = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, nodes_vector: Any, edge_set: Any): ...

class Flatten(layers.Layer):
    start_axis: Any = ...
    stop_axis: Any = ...
    def __init__(self, start_axis: int = ..., stop_axis: int = ...) -> None: ...
    def forward(self, input: Tensor) -> Tensor: ...
    __call__ = forward
