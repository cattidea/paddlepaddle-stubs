from __future__ import annotations

from collections import defaultdict as defaultdict
from typing import Any

from .process_mesh import ProcessMesh as ProcessMesh

def get_tensor_dist_attr_field_keys(): ...
def get_op_dist_attr_field_keys(): ...
def append_op_input_suffix(name: Any): ...
def append_op_output_suffix(name: Any): ...

class TensorDistributedAttribute:
    def __init__(self) -> None: ...
    @property
    def process_mesh(self): ...
    @process_mesh.setter
    def process_mesh(self, process_mesh: Any) -> None: ...
    @property
    def dims_mapping(self): ...
    @dims_mapping.setter
    def dims_mapping(self, dims_mapping: Any) -> None: ...
    @property
    def shard_sizes(self): ...
    @shard_sizes.setter
    def shard_sizes(self, shard_sizes: Any) -> None: ...
    @property
    def device_placement(self): ...
    @device_placement.setter
    def device_placement(self, device_placement: Any) -> None: ...
    def init(self, dist_attr: Any) -> None: ...
    def is_annotated(self, dist_attr_field_name: Any): ...
    def mark_annotated(self, dist_attr_field_name: Any) -> None: ...
    def mark_annotated_as(self, dist_attr: Any) -> None: ...
    def clear_annotated(self) -> None: ...

class OperatorDistributedAttribute:
    def __init__(self) -> None: ...
    @property
    def process_mesh(self): ...
    @process_mesh.setter
    def process_mesh(self, process_mesh: Any) -> None: ...
    @property
    def op_type(self): ...
    @op_type.setter
    def op_type(self, op_type: Any) -> None: ...
    @property
    def impl_type(self): ...
    @impl_type.setter
    def impl_type(self, impl_type: Any) -> None: ...
    @property
    def impl_idx(self): ...
    @impl_idx.setter
    def impl_idx(self, impl_idx: Any) -> None: ...
    @property
    def is_recompute(self): ...
    @is_recompute.setter
    def is_recompute(self, is_recompute: Any) -> None: ...
    @property
    def inputs_dist_attrs(self): ...
    @property
    def outputs_dist_attrs(self): ...
    def get_input_dist_attr(self, name: Any): ...
    def set_input_dist_attr(self, name: Any, dist_attr: Any) -> None: ...
    def get_output_dist_attr(self, name: Any): ...
    def set_output_dist_attr(self, name: Any, dist_attr: Any) -> None: ...
    def get_input_dims_mapping(self, name: Any): ...
    def set_input_dims_mapping(self, name: Any, dims_mapping: Any) -> None: ...
    def get_output_dims_mapping(self, name: Any): ...
    def set_output_dims_mapping(self, name: Any, dims_mapping: Any) -> None: ...
    def init(self, dist_attr: Any) -> None: ...
    def is_annotated(self, attr_name: Any): ...
    def mark_annotated(self, attr_name: Any) -> None: ...
    def mark_annotated_as(self, dist_attr: Any) -> None: ...
    def clear_annotated(self) -> None: ...
    def is_annotated_input_dims_mapping(self, name: Any): ...
    def is_annotated_output_dims_mapping(self, name: Any): ...
