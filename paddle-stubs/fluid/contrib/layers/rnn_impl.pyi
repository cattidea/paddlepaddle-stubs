from __future__ import annotations

from typing import Any, Optional

from paddle.fluid.dygraph import Layer

class BasicGRUUnit(Layer):
    def __init__(
        self,
        name_scope: Any,
        hidden_size: Any,
        param_attr: Any | None = ...,
        bias_attr: Any | None = ...,
        gate_activation: Any | None = ...,
        activation: Any | None = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, input: Any, pre_hidden: Any): ...

def basic_gru(
    input: Any,
    init_hidden: Any,
    hidden_size: Any,
    num_layers: int = ...,
    sequence_length: Any | None = ...,
    dropout_prob: float = ...,
    bidirectional: bool = ...,
    batch_first: bool = ...,
    param_attr: Any | None = ...,
    bias_attr: Any | None = ...,
    gate_activation: Any | None = ...,
    activation: Any | None = ...,
    dtype: str = ...,
    name: str = ...,
): ...
def basic_lstm(
    input: Any,
    init_hidden: Any,
    init_cell: Any,
    hidden_size: Any,
    num_layers: int = ...,
    sequence_length: Any | None = ...,
    dropout_prob: float = ...,
    bidirectional: bool = ...,
    batch_first: bool = ...,
    param_attr: Any | None = ...,
    bias_attr: Any | None = ...,
    gate_activation: Any | None = ...,
    activation: Any | None = ...,
    forget_bias: float = ...,
    dtype: str = ...,
    name: str = ...,
): ...

class BasicLSTMUnit(Layer):
    def __init__(
        self,
        name_scope: Any,
        hidden_size: Any,
        param_attr: Any | None = ...,
        bias_attr: Any | None = ...,
        gate_activation: Any | None = ...,
        activation: Any | None = ...,
        forget_bias: float = ...,
        dtype: str = ...,
    ) -> None: ...
    def forward(self, input: Any, pre_hidden: Any, pre_cell: Any): ...
