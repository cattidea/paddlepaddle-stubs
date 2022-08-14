from __future__ import annotations

from typing import Any, Optional

from paddle.nn import Layer, LayerList
from typing_extensions import Literal

from ..._typing import DTypeLike, ShapeLike, Tensor
from ...framework import ParamAttr

RNNDirection = Literal["forward", "bidirect", "bidirectional"]

class RNNCellBase(Layer):
    shape: Any = ...
    def get_initial_states(
        self,
        batch_ref: Tensor,
        shape: Optional[ShapeLike] = ...,
        dtype: Optional[DTypeLike] = ...,
        init_value: float = ...,
        batch_dim_idx: int = ...,
    ) -> Tensor | list[Tensor]: ...
    @property
    def state_shape(self) -> None: ...
    @property
    def state_dtype(self) -> None: ...

class SimpleRNNCell(RNNCellBase):
    weight_ih: Any = ...
    weight_hh: Any = ...
    bias_ih: Any = ...
    bias_hh: Any = ...
    input_size: Any = ...
    hidden_size: Any = ...
    activation: Any = ...
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: Literal["relu", "tanh"] = ...,
        weight_ih_attr: Optional[ParamAttr] = ...,
        weight_hh_attr: Optional[ParamAttr] = ...,
        bias_ih_attr: Optional[ParamAttr] = ...,
        bias_hh_attr: Optional[ParamAttr] = ...,
        name: Optional[str] = ...,
    ) -> None: ...
    def forward(
        self,
        inputs: Tensor,
        states: Optional[Tensor] = ...,
    ) -> tuple[Tensor, Tensor]: ...
    __call__ = forward
    @property
    def state_shape(self) -> tuple[int]: ...

class LSTMCell(RNNCellBase):
    weight_ih: Any = ...
    weight_hh: Any = ...
    bias_ih: Any = ...
    bias_hh: Any = ...
    hidden_size: Any = ...
    input_size: Any = ...
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        weight_ih_attr: Optional[ParamAttr] = ...,
        weight_hh_attr: Optional[ParamAttr] = ...,
        bias_ih_attr: Optional[ParamAttr] = ...,
        bias_hh_attr: Optional[ParamAttr] = ...,
        name: Optional[str] = ...,
    ) -> None: ...
    def forward(
        self,
        inputs: Tensor,
        states: Optional[Tensor] = ...,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]: ...
    __call__ = forward
    @property
    def state_shape(self) -> tuple[tuple[int], tuple[int]]: ...

class GRUCell(RNNCellBase):
    weight_ih: Any = ...
    weight_hh: Any = ...
    bias_ih: Any = ...
    bias_hh: Any = ...
    hidden_size: Any = ...
    input_size: Any = ...
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        weight_ih_attr: Optional[ParamAttr] = ...,
        weight_hh_attr: Optional[ParamAttr] = ...,
        bias_ih_attr: Optional[ParamAttr] = ...,
        bias_hh_attr: Optional[ParamAttr] = ...,
        name: Optional[str] = ...,
    ) -> None: ...
    def forward(self, inputs: Tensor, states: Optional[Tensor] = ...) -> tuple[Tensor, Tensor]: ...
    __call__ = forward
    @property
    def state_shape(self) -> tuple[int]: ...

class RNN(Layer):
    cell: Any = ...
    is_reverse: Any = ...
    time_major: Any = ...
    def __init__(
        self,
        cell: RNNCellBase,
        is_reverse: bool = ...,
        time_major: bool = ...,
    ) -> None: ...
    def forward(
        self,
        inputs: Tensor,
        initial_states: Optional[Tensor] = ...,
        sequence_length: Optional[Tensor] = ...,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]: ...
    __call__ = forward

class BiRNN(Layer):
    cell_fw: Any = ...
    cell_bw: Any = ...
    time_major: Any = ...
    def __init__(
        self,
        cell_fw: RNNCellBase,
        cell_bw: RNNCellBase,
        time_major: bool = ...,
    ) -> None: ...
    def forward(
        self,
        inputs: Tensor,
        initial_states: Optional[Tensor] = ...,
        sequence_length: Optional[Tensor] = ...,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]: ...
    __call__ = forward

class RNNBase(LayerList):
    mode: Any = ...
    input_size: Any = ...
    hidden_size: Any = ...
    dropout: Any = ...
    num_directions: Any = ...
    time_major: Any = ...
    num_layers: Any = ...
    state_components: Any = ...
    could_use_cudnn: bool = ...
    def __init__(
        self,
        mode: Literal["RNN_TANH", "RNN_RELU", "LSTM", "GRU"],
        input_size: int,
        hidden_size: int,
        num_layers: int = ...,
        direction: RNNDirection = ...,
        time_major: bool = ...,
        dropout: float = ...,
        weight_ih_attr: Optional[ParamAttr] = ...,
        weight_hh_attr: Optional[ParamAttr] = ...,
        bias_ih_attr: Optional[ParamAttr] = ...,
        bias_hh_attr: Optional[ParamAttr] = ...,
    ) -> None: ...
    def flatten_parameters(self) -> None: ...
    def forward(
        self,
        inputs: Tensor,
        initial_states: Optional[Tensor] = ...,
        sequence_length: Optional[Tensor] = ...,
    ) -> tuple[Tensor, Tensor]: ...
    __call__ = forward

class SimpleRNN(RNNBase):
    activation: Any = ...
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = ...,
        direction: RNNDirection = ...,
        time_major: bool = ...,
        dropout: float = ...,
        activation: str = ...,
        weight_ih_attr: Optional[ParamAttr] = ...,
        weight_hh_attr: Optional[ParamAttr] = ...,
        bias_ih_attr: Optional[ParamAttr] = ...,
        bias_hh_attr: Optional[ParamAttr] = ...,
        name: Optional[str] = ...,
    ) -> None: ...

class LSTM(RNNBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = ...,
        direction: RNNDirection = ...,
        time_major: bool = ...,
        dropout: float = ...,
        weight_ih_attr: Optional[ParamAttr] = ...,
        weight_hh_attr: Optional[ParamAttr] = ...,
        bias_ih_attr: Optional[ParamAttr] = ...,
        bias_hh_attr: Optional[ParamAttr] = ...,
        name: Optional[str] = ...,
    ) -> None: ...

class GRU(RNNBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = ...,
        direction: RNNDirection = ...,
        time_major: bool = ...,
        dropout: float = ...,
        weight_ih_attr: Optional[ParamAttr] = ...,
        weight_hh_attr: Optional[ParamAttr] = ...,
        bias_ih_attr: Optional[ParamAttr] = ...,
        bias_hh_attr: Optional[ParamAttr] = ...,
        name: Optional[str] = ...,
    ) -> None: ...
