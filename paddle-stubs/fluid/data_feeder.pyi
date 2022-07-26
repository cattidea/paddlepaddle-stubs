from __future__ import annotations

from typing import Any, Optional

from .._typing import DTypeLike, ShapeLike, Tensor, _DTypeString
from .framework import Variable

def convert_dtype(dtype: DTypeLike) -> str: ...
def check_variable_and_dtype(
    input: Tensor,
    input_name: str,
    expected_dtype: DTypeLike,
    op_name: str,
    extra_message: str = ...,
) -> None: ...
def check_type(
    input: Tensor,
    input_name: str,
    expected_type: type[Tensor],
    op_name: str,
    extra_message: str = ...,
) -> None: ...
def check_dtype(
    input_dtype: DTypeLike,
    input_name: str,
    expected_dtype: DTypeLike,
    op_name: str,
    extra_message: str = ...,
) -> None: ...
def check_shape(
    shape: ShapeLike,
    op_name: str,
    expected_shape_type: tuple[type[list] | type[tuple] | type[Variable], ...] = ...,
    expected_element_type: tuple[type[list] | type[tuple] | type[Variable], ...] = ...,
    expected_tensor_dtype: _DTypeString = ...,
) -> None: ...

class DataToLoDTensorConverter:
    place: Any = ...
    lod_level: Any = ...
    shape: Any = ...
    dtype: Any = ...
    def __init__(self, place: Any, lod_level: Any, shape: Any, dtype: Any) -> None: ...
    def feed(self, data: Any) -> None: ...
    def done(self): ...

class BatchedTensorProvider:
    place: Any = ...
    batch_size: Any = ...
    generator: Any = ...
    converters: Any = ...
    drop_last: Any = ...
    def __init__(self, feed_list: Any, place: Any, batch_size: Any, generator: Any, drop_last: Any) -> None: ...
    def __call__(self) -> None: ...

class DataFeeder:
    feed_dtypes: Any = ...
    feed_names: Any = ...
    feed_shapes: Any = ...
    feed_lod_level: Any = ...
    place: Any = ...
    def __init__(self, feed_list: Any, place: Any, program: Any | None = ...) -> None: ...
    def feed(self, iterable: Any): ...
    def feed_parallel(self, iterable: Any, num_places: Any | None = ...) -> None: ...
    def decorate_reader(self, reader: Any, multi_devices: Any, num_places: Any | None = ..., drop_last: bool = ...): ...
