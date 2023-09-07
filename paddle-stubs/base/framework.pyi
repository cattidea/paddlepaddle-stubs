from __future__ import annotations

from typing import Any, Callable, Optional

from .._typing import NPUPlace, _DTypeNumpy, dtype
from . import core

def in_dygraph_mode(): ...
def ipu_shard_guard(index: Any | None = ..., stage: Any | None = ...) -> None: ...
def require_version(min_version: Any, max_version: Any | None = ...): ...
def is_compiled_with_xpu(): ...
def is_compiled_with_npu(): ...
def is_compiled_with_cinn(): ...
def is_compiled_with_cuda(): ...
def is_compiled_with_rocm(): ...
def cuda_places(device_ids: Any | None = ...): ...
def xpu_places(device_ids: Any | None = ...): ...
def npu_places(device_ids: list[int] | None = ...) -> list[NPUPlace]: ...
def cpu_places(device_count: Any | None = ...): ...
def cuda_pinned_places(device_count: Any | None = ...): ...
def mlu_places(device_ids: Any | None = ...): ...
def dygraph_not_support(func: Callable[..., Any]) -> Callable[..., Any]: ...
def dygraph_only(func: Callable[..., Any]) -> Callable[..., Any]: ...
def static_only(func: Callable[..., Any]) -> Callable[..., Any]: ...
def fake_interface_only(func: Callable[..., Any]) -> Callable[..., Any]: ...
def convert_np_dtype_to_dtype_(np_dtype: _DTypeNumpy) -> dtype: ...
def disable_signal_handler() -> None: ...
def monkey_patch_math_varbase() -> None: ...

class NameScope:
    def __init__(self, name: str = ..., parent: Any | None = ...) -> None: ...
    def child(self, prefix: Any): ...
    def parent(self): ...
    def name(self): ...

def name_scope(prefix: Any | None = ...) -> None: ...

class VariableMetaClass(type):
    @classmethod
    def __instancecheck__(cls, instance: Any): ...

class ParameterMetaClass(VariableMetaClass):
    @classmethod
    def __instancecheck__(cls, instance: Any): ...

class Variable(metaclass=VariableMetaClass):
    block: Any = ...
    belong_to_optimizer: Any = ...
    error_clip: Any = ...
    desc: Any = ...
    op: Any = ...
    is_data: Any = ...
    def __init__(
        self,
        block: Any,
        type: Any = ...,
        name: str | None = ...,
        shape: Any | None = ...,
        dtype: Any | None = ...,
        lod_level: Any | None = ...,
        capacity: Any | None = ...,
        persistable: Any | None = ...,
        error_clip: Any | None = ...,
        stop_gradient: bool = ...,
        is_data: bool = ...,
        need_check_feed: bool = ...,
        belong_to_optimizer: bool = ...,
        **kwargs: Any,
    ) -> None: ...
    def detach(self): ...
    def numpy(self) -> None: ...
    def backward(self, retain_graph: bool = ...) -> None: ...
    def gradient(self) -> None: ...
    def clear_gradient(self) -> None: ...
    def register_hook(self, hook: Any) -> None: ...
    def to_string(self, throw_on_error: Any, with_details: bool = ...): ...
    def element_size(self): ...
    @property
    def stop_gradient(self): ...
    @stop_gradient.setter
    def stop_gradient(self, s: Any) -> None: ...
    @property
    def persistable(self): ...
    @persistable.setter
    def persistable(self, p: Any) -> None: ...
    @property
    def is_parameter(self): ...
    @is_parameter.setter
    def is_parameter(self, p: Any) -> None: ...
    @property
    def name(self): ...
    @property
    def grad_name(self): ...
    @name.setter
    def name(self, new_name: Any) -> None: ...
    @property
    def shape(self): ...
    @property
    def dtype(self): ...
    @property
    def lod_level(self): ...
    @property
    def type(self): ...
    @property
    def T(self): ...
    def clone(self): ...
    def __getitem__(self, item: Any): ...
    def __setitem__(self, item: Any, value: Any): ...
    def get_value(self, scope: Any | None = ...): ...
    def set_value(self, value: Any, scope: Any | None = ...) -> None: ...
    def size(self): ...
    @property
    def attr_names(self): ...
    @property
    def process_mesh(self): ...
    @property
    def shard_mask(self): ...
    @property
    def offload_device(self): ...

class OpProtoHolder:
    @classmethod
    def instance(cls): ...
    op_proto_map: Any = ...
    def __init__(self) -> None: ...
    def get_op_proto(self, type: Any): ...
    def update_op_proto(self): ...
    @staticmethod
    def generated_op_attr_names(): ...

class Operator:
    OP_WITHOUT_KERNEL_SET: Any = ...
    attrs: Any = ...
    block: Any = ...
    desc: Any = ...
    def __init__(
        self,
        block: Any,
        desc: Any,
        type: Any | None = ...,
        inputs: Any | None = ...,
        outputs: Any | None = ...,
        attrs: Any | None = ...,
    ): ...
    def to_string(self, throw_on_error: Any): ...
    @property
    def type(self): ...
    def input(self, name: Any): ...
    @property
    def input_names(self): ...
    @property
    def input_arg_names(self): ...
    @property
    def output_arg_names(self): ...
    def output(self, name: Any): ...
    @property
    def output_names(self): ...
    @property
    def idx(self): ...
    def has_attr(self, name: Any): ...
    def attr_type(self, name: Any): ...
    @property
    def attr_names(self): ...
    def attr(self, name: Any): ...
    def all_attrs(self): ...
    @property
    def process_mesh(self): ...
    def dims_mapping(self, name: Any): ...
    @property
    def pipeline_stage(self): ...

class Block:
    desc: Any = ...
    vars: Any = ...
    ops: Any = ...
    program: Any = ...
    removed_vars: Any = ...
    def __init__(self, program: Any, idx: Any) -> None: ...
    def to_string(self, throw_on_error: Any, with_details: bool = ...): ...
    @property
    def parent_idx(self): ...
    @property
    def forward_block_idx(self): ...
    @property
    def backward_block_idx(self): ...
    @property
    def idx(self): ...
    def var(self, name: Any): ...
    def all_parameters(self): ...
    def iter_parameters(self): ...
    def create_var(self, *args: Any, **kwargs: Any): ...
    def has_var(self, name: Any): ...
    def create_parameter(self, *args: Any, **kwargs: Any): ...
    def append_op(self, *args: Any, **kwargs: Any): ...

class IrNode:
    node: Any = ...
    def __init__(self, node: Any) -> None: ...
    def name(self): ...
    def node_type(self): ...
    def var(self): ...
    def op(self): ...
    def id(self): ...
    def is_op(self): ...
    def is_var(self): ...
    def is_ctrl_var(self): ...
    def clear_inputs(self) -> None: ...
    def remove_input_by_id(self, node_id: Any) -> None: ...
    def remove_input(self, node: Any) -> None: ...
    def append_input(self, node: Any) -> None: ...
    def clear_outputs(self) -> None: ...
    def remove_output_by_id(self, node_id: Any) -> None: ...
    def remove_output(self, node: Any) -> None: ...
    def append_output(self, node: Any) -> None: ...
    @property
    def inputs(self): ...
    @property
    def outputs(self): ...

class IrVarNode(IrNode):
    node: Any = ...
    def __init__(self, node: Any) -> None: ...
    def set_shape(self, shape: Any) -> None: ...
    def persistable(self): ...
    def type(self): ...
    def dtype(self): ...
    def shape(self): ...
    @property
    def inputs(self): ...
    @property
    def outputs(self): ...

class IrOpNode(IrNode):
    node: Any = ...
    def __init__(self, node: Any) -> None: ...
    def rename_input(self, old_input_name: Any, new_input_name: Any) -> None: ...
    def rename_output(self, old_output_name: Any, new_output_name: Any) -> None: ...
    def input(self, name: Any): ...
    def output(self, name: Any): ...
    def set_type(self, new_type: Any): ...
    def set_attr(self, name: Any, val: Any) -> None: ...
    def input_arg_names(self): ...
    def output_arg_names(self): ...
    @property
    def inputs(self): ...
    @property
    def outputs(self): ...

class IrGraph:
    graph: Any = ...
    def __init__(self, graph: Any, for_test: bool = ...) -> None: ...
    def clone(self): ...
    def is_test(self): ...
    def all_nodes(self): ...
    def all_var_nodes(self): ...
    def all_persistable_nodes(self): ...
    def all_op_nodes(self): ...
    def all_sub_graphs(self, for_test: bool = ...): ...
    def get_sub_graph(self, i: Any, for_test: bool = ...): ...
    def create_persistable_node(self, name: Any, var_type: Any, shape: Any, var_dtype: Any): ...
    def create_var_node(self, name: Any, var_type: Any, shape: Any, var_dtype: Any): ...
    def create_control_dep_var(self): ...
    def create_var_node_from_desc(self, var_desc: Any): ...
    def create_op_node(self, op_type: Any, attrs: Any, inputs: Any, outputs: Any): ...
    def create_op_node_from_desc(self, op_desc: Any): ...
    def update_input_link(self, old_input_node: Any, new_input_node: Any, op_node: Any) -> None: ...
    def update_output_link(self, old_output_node: Any, new_output_node: Any, op_node: Any) -> None: ...
    def link_to(self, node_in: Any, node_out: Any) -> None: ...
    def safe_remove_nodes(self, remove_nodes: Any) -> None: ...
    def resolve_hazard(self) -> None: ...
    def has_circle(self): ...
    def graph_num(self): ...
    def topology_sort(self): ...
    def build_adjacency_list(self): ...
    def draw(self, save_path: Any, name: Any, marked_nodes: Any | None = ..., remove_ctr_var: bool = ...) -> None: ...
    def to_program(self): ...

class Program:
    desc: Any = ...
    blocks: Any = ...
    current_block_idx: int = ...
    def __init__(self) -> None: ...
    def global_seed(self, seed: int = ...) -> None: ...
    def to_string(self, throw_on_error: Any, with_details: bool = ...): ...
    def clone(self, for_test: bool = ...): ...
    @staticmethod
    def parse_from_string(binary_str: Any): ...
    @property
    def random_seed(self): ...
    @property
    def num_blocks(self): ...
    @random_seed.setter
    def random_seed(self, seed: Any) -> None: ...
    def global_block(self): ...
    def block(self, index: Any): ...
    def current_block(self): ...
    def list_vars(self) -> None: ...
    def all_parameters(self): ...
    def state_dict(self, mode: str = ..., scope: Any | None = ...): ...
    def set_state_dict(self, state_dict: Any, scope: Any | None = ...) -> None: ...

class Parameter(Variable, metaclass=ParameterMetaClass):
    trainable: Any = ...
    optimize_attr: Any = ...
    regularizer: Any = ...
    do_model_average: Any = ...
    need_clip: Any = ...
    is_distributed: bool = ...
    is_parameter: bool = ...
    def __init__(self, block: Any, shape: Any, dtype: Any, type: Any = ..., **kwargs: Any) -> None: ...
    def to_string(self, throw_on_error: Any, with_details: bool = ...): ...

class ParamBase(core.VarBase):
    stop_gradient: Any = ...
    optimize_attr: Any = ...
    regularizer: Any = ...
    do_model_average: Any = ...
    need_clip: Any = ...
    is_distributed: Any = ...
    def __init__(self, shape: Any, dtype: Any, **kwargs: Any) -> None: ...
    @property
    def trainable(self): ...
    @trainable.setter
    def trainable(self, trainable: Any) -> None: ...
    def __deepcopy__(self, memo: Any): ...

class EagerParamBase:
    stop_gradient: Any = ...
    optimize_attr: Any = ...
    regularizer: Any = ...
    do_model_average: Any = ...
    need_clip: Any = ...
    is_distributed: Any = ...
    def __init__(self, shape: Any, dtype: Any, **kwargs: Any) -> None: ...
    @property
    def trainable(self): ...
    @trainable.setter
    def trainable(self, trainable: Any) -> None: ...
    def __deepcopy__(self, memo: Any): ...

def default_startup_program(): ...
def default_main_program(): ...
def program_guard(main_program: Any, startup_program: Any | None = ...) -> None: ...
def device_guard(device: Any | None = ...) -> None: ...
def set_flags(flags: Any) -> None: ...
def get_flags(flags: Any): ...

# Names in __all__ with no definition:
#   _non_static_mode