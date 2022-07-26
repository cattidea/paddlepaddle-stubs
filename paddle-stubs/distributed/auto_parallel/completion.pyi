from __future__ import annotations

from copy import deepcopy as deepcopy
from typing import Any

from paddle.fluid import framework as framework

from .dist_attribute import OperatorDistributedAttribute as OperatorDistributedAttribute
from .dist_attribute import TensorDistributedAttribute as TensorDistributedAttribute
from .dist_context import (
    get_default_distributed_context as get_default_distributed_context,
)
from .dist_op import DistributedOperator as DistributedOperator
from .dist_tensor import DistributedTensor as DistributedTensor
from .operators import (
    find_best_compatible_distributed_operator_impl as find_best_compatible_distributed_operator_impl,
)
from .process_mesh import ProcessMesh as ProcessMesh
from .utils import print_program_with_dist_attr as print_program_with_dist_attr

def compute_compatible_process_mesh(process_mesh_list: Any): ...
def compute_compatible_dim_mapping(dim_mapping_list: Any): ...
def compute_compatible_dims_mapping(dims_mapping_list: Any): ...
def merge_process_mesh_two(pm1: Any, pm2: Any): ...

class Completer:
    def __init__(self, dist_context: Any) -> None: ...
    def complete_forward_annotation(self, serial_main_program: Any): ...
    def complete_backward_annotation(self, serial_main_program: Any): ...
    def complete_update_annotation(self, serial_main_program: Any) -> None: ...
