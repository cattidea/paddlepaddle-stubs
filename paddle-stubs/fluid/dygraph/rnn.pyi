from __future__ import annotations

from typing import Any, Optional

from . import Layer

class LSTMCell(Layer):
    def __init__(
        self,
        hidden_size: Any,
        input_size: Any,
        param_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        gate_activation: Optional[Any] = ...,
        activation: Optional[Any] = ...,
        forget_bias: float = ...,
        use_cudnn_impl: bool = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, input: Any, pre_hidden: Any, pre_cell: Any): ...

class GRUCell(Layer):
    def __init__(
        self,
        hidden_size: Any,
        input_size: Any,
        param_attr: Optional[Any] = ...,
        bias_attr: Optional[Any] = ...,
        gate_activation: Optional[Any] = ...,
        activation: Optional[Any] = ...,
        use_cudnn_impl: bool = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, input: Any, pre_hidden: Any): ...
