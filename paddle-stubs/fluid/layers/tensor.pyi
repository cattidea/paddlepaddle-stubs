from __future__ import annotations

from typing import Any, Optional

def create_tensor(dtype: Any, name: str | None = ..., persistable: bool = ...): ...
def create_parameter(
    shape: Any,
    dtype: Any,
    name: str | None = ...,
    attr: Any | None = ...,
    is_bias: bool = ...,
    default_initializer: Any | None = ...,
): ...
def create_global_var(
    shape: Any, value: Any, dtype: Any, persistable: bool = ..., force_cpu: bool = ..., name: str | None = ...
): ...
def cast(x: Any, dtype: Any): ...
def concat(input: Any, axis: int = ..., name: str | None = ...): ...
def tensor_array_to_tensor(input: Any, axis: int = ..., name: str | None = ..., use_stack: bool = ...): ...
def sums(input: Any, out: Any | None = ...): ...
def assign(input: Any, output: Any | None = ...): ...
def fill_constant(
    shape: Any, dtype: Any, value: Any, force_cpu: bool = ..., out: Any | None = ..., name: str | None = ...
): ...
def fill_constant_batch_size_like(
    input: Any,
    shape: Any,
    dtype: Any,
    value: Any,
    input_dim_idx: int = ...,
    output_dim_idx: int = ...,
    force_cpu: bool = ...,
): ...
def argmin(x: Any, axis: int = ...): ...
def argmax(x: Any, axis: int = ...): ...
def argsort(input: Any, axis: int = ..., descending: bool = ..., name: str | None = ...): ...
def ones(shape: Any, dtype: Any, force_cpu: bool = ...): ...
def zeros(shape: Any, dtype: Any, force_cpu: bool = ..., name: str | None = ...): ...
def reverse(x: Any, axis: Any): ...
def has_inf(x: Any): ...
def has_nan(x: Any): ...
def isfinite(x: Any): ...
def range(start: Any, end: Any, step: Any, dtype: Any, name: str | None = ...): ...
def linspace(start: Any, stop: Any, num: Any, dtype: Any | None = ..., name: str | None = ...): ...
def zeros_like(x: Any, out: Any | None = ...): ...
def diag(diagonal: Any): ...
def eye(
    num_rows: Any,
    num_columns: Any | None = ...,
    batch_shape: Any | None = ...,
    dtype: str = ...,
    name: str | None = ...,
): ...
def ones_like(x: Any, out: Any | None = ...): ...
def triu(input: Any, diagonal: int = ..., name: str | None = ...): ...
