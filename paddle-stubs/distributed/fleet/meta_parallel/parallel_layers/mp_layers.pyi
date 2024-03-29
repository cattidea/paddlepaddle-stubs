from __future__ import annotations

from typing import Any, Optional

from paddle import framework as framework
from paddle.autograd import PyLayer as PyLayer
from paddle.base.dygraph.layers import Layer

from .random import get_rng_state_tracker as get_rng_state_tracker

class VocabParallelEmbedding(Layer):
    model_parallel_group: Any = ...
    world_size: Any = ...
    rank: Any = ...
    origin_num_embeddings: Any = ...
    is_mp: Any = ...
    vocab_start_index: Any = ...
    weight: Any = ...
    def __init__(
        self, num_embeddings: Any, embedding_dim: Any, weight_attr: Any | None = ..., name: str | None = ...
    ) -> None: ...
    def forward(self, x: Any): ...

class ColumnParallelLinear(Layer):
    model_parallel_group: Any = ...
    world_size: Any = ...
    is_mp: Any = ...
    gather_output: Any = ...
    output_size_per_partition: Any = ...
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        in_features: Any,
        out_features: Any,
        weight_attr: Any | None = ...,
        has_bias: Any | None = ...,
        gather_output: bool = ...,
        name: str | None = ...,
    ) -> None: ...
    def forward(self, x: Any): ...

class RowParallelLinear(Layer):
    in_features: Any = ...
    out_features: Any = ...
    input_is_parallel: Any = ...
    model_parallel_group: Any = ...
    world_size: Any = ...
    rank: Any = ...
    is_mp: Any = ...
    input_size_per_partition: Any = ...
    weight: Any = ...
    bias: Any = ...
    def __init__(
        self,
        in_features: Any,
        out_features: Any,
        weight_attr: Any | None = ...,
        has_bias: bool = ...,
        input_is_parallel: bool = ...,
        name: str | None = ...,
    ) -> None: ...
    def forward(self, x: Any): ...

class ParallelCrossEntropy(Layer):
    name: Any = ...
    model_parallel_group: Any = ...
    world_size: Any = ...
    rank: Any = ...
    def __init__(self, name: str | None = ...) -> None: ...
    def forward(self, input: Any, label: Any): ...
