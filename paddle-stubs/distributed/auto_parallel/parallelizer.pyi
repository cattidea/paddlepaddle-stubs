from __future__ import annotations

from typing import Any, Optional

from paddle.distributed.fleet import cloud_utils as cloud_utils

from .cluster import Cluster as Cluster
from .completion import Completer as Completer
from .dist_context import DistributedContext as DistributedContext
from .dist_context import (
    get_default_distributed_context as get_default_distributed_context,
)
from .dist_context import (
    set_default_distributed_context as set_default_distributed_context,
)
from .dist_op import DistributedOperator as DistributedOperator
from .dist_tensor import DistributedTensor as DistributedTensor
from .mapper import mapping as mapping
from .partitioner import Partitioner as Partitioner
from .planner import Planner as Planner
from .process_group import ProcessGroup as ProcessGroup
from .process_group import get_all_process_groups as get_all_process_groups
from .process_group import get_process_group as get_process_group
from .process_group import get_world_process_group as get_world_process_group
from .reshard import Resharder as Resharder
from .utils import SerialProgramInfo as SerialProgramInfo
from .utils import make_data_unshard as make_data_unshard
from .utils import print_program_with_dist_attr as print_program_with_dist_attr
from .utils import set_grad_var_shape as set_grad_var_shape

class AutoParallelizer:
    def __init__(self, fleet: Any) -> None: ...
    def parallelize(
        self,
        loss: Any,
        startup_program: Any,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
        callbacks: Any | None = ...,
    ): ...
    def __deepcopy__(self, memo: Any): ...
