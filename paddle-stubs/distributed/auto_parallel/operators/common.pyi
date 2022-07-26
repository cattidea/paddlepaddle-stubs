from __future__ import annotations

import abc
from typing import Any

from ..dist_attribute import (
    OperatorDistributedAttribute as OperatorDistributedAttribute,
)

BACKWARD_ONLY_DIST_OPS: Any

def is_elementwise_op(op_type: Any): ...

class DistributedOperatorImplContainer:
    def __init__(self, op_type: Any) -> None: ...
    @property
    def type(self): ...
    @type.setter
    def type(self, op_type: Any) -> None: ...
    @property
    def impls(self): ...
    def register_impl(self, dist_impl: Any) -> None: ...
    def get_impl(self, impl_idx: Any): ...
    def get_input_compatible_impls(self, dist_op: Any): ...
    def get_output_compatible_impls(self, dist_op: Any): ...
    def get_compatible_impls(self, dist_op: Any): ...

class DistributedOperatorImpl(abc.ABC, metaclass=abc.ABCMeta):
    def __init__(self, name: Any) -> None: ...
    @property
    def name(self): ...
    @name.setter
    def name(self, name: Any) -> None: ...
    @property
    def type(self): ...
    @type.setter
    def type(self, op_type: Any) -> None: ...
    @property
    def idx(self): ...
    @idx.setter
    def idx(self, impl_idx: Any) -> None: ...
    @abc.abstractmethod
    def is_input_compatible(self, dist_op: Any) -> Any: ...
    @abc.abstractmethod
    def is_output_compatible(self, dist_op: Any) -> Any: ...
    @abc.abstractmethod
    def is_auto_compatible(self, dist_op: Any) -> Any: ...
    @staticmethod
    @abc.abstractmethod
    def forward(dist_ctx: Any, *args: Any, **kwargs: Any) -> Any: ...
    @staticmethod
    @abc.abstractmethod
    def backward(dist_ctx: Any, *grad_outputs: Any, **kwargs: Any) -> Any: ...
    def update_dims_mapping(self, dist_op: Any) -> None: ...

def register_distributed_operator_impl_container(container: Any) -> None: ...
def get_distributed_operator_impl_container(op_type: Any): ...
def register_distributed_operator_impl(op_type: Any, dist_impl: Any) -> None: ...
def find_best_compatible_distributed_operator_impl(dist_op: Any, fwd: bool = ...): ...
def is_parameter_related(varname: Any, block: Any): ...
def infer_shape(block: Any, src_var: Any, src_var_dist_attr: Any, op_input_dist_attr: Any): ...
def set_comm_op_dist_attr_for_program(new_op: Any, process_mesh: Any, tensor_dist_attr: Any, ctx: Any) -> None: ...
def naive_copy_op_dist_attr_for_program(new_op: Any, ref_op: Any, ctx: Any) -> None: ...
