from __future__ import annotations

from typing import Any, Optional

from .dist_attribute import TensorDistributedAttribute as TensorDistributedAttribute
from .dist_attribute import (
    get_tensor_dist_attr_field_keys as get_tensor_dist_attr_field_keys,
)

class DistributedTensor:
    @staticmethod
    def get_local_sizes(
        global_sizes: Any,
        dims_mapping: Any,
        topology: Any,
        processes: Any,
        rank: Optional[Any] = ...,
        shard_sizes: Optional[Any] = ...,
    ): ...
    @staticmethod
    def get_local_offsets(
        global_sizes: Any, dims_mapping: Any, topology: Any, processes: Any, rank: Any, shard_sizes: Optional[Any] = ...
    ): ...
    @staticmethod
    def get_global_sizes(
        local_sizes: Any,
        dims_mapping: Any,
        topology: Any,
        processes: Any,
        rank: Optional[Any] = ...,
        shard_sizes: Optional[Any] = ...,
    ): ...
    @staticmethod
    def get_local_shard(
        global_sizes: Any, dims_mapping: Any, topology: Any, processes: Any, rank: Any, shard_sizes: Optional[Any] = ...
    ): ...
    def __init__(
        self, serial_tensor: Any, dist_attr: Optional[Any] = ..., dist_context: Optional[Any] = ...
    ) -> None: ...
    @property
    def serial_tensor(self): ...
    @property
    def dist_attr(self): ...
    @property
    def dist_context(self): ...
    @dist_attr.setter
    def dist_attr(self, dist_attr: Any) -> None: ...
    def validate_dist_attr(self): ...
    def local_sizes(self, rank: Optional[Any] = ...): ...
    def local_offsets(self, rank: Optional[Any] = ...): ...
    def global_sizes(self): ...
    def local_shard(self, rank: Optional[Any] = ...): ...
    def new_local_tensor(self, block: Optional[Any] = ..., rank: Optional[Any] = ..., name: Optional[Any] = ...): ...
    def local_tensor(self, rank: Optional[Any] = ...): ...
    def __deepcopy__(self, memo: Any): ...
