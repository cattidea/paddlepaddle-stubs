from __future__ import annotations

from typing import Any, Optional

from paddle.fluid import core as core
from paddle.metric import Metric as Metric

from .cluster import Cluster as Cluster
from .completion import Completer as Completer
from .dist_context import DistributedContext as DistributedContext
from .dist_context import (
    get_default_distributed_context as get_default_distributed_context,
)
from .dist_loader import NonIterableGeneratorLoader as NonIterableGeneratorLoader
from .dist_op import DistributedOperator as DistributedOperator
from .dist_saver import DistributedSaver as DistributedSaver
from .mapper import mapping as mapping
from .partitioner import Partitioner as Partitioner
from .planner import Planner as Planner
from .process_group import get_all_process_groups as get_all_process_groups
from .process_group import get_world_process_group as get_world_process_group
from .reshard import Resharder as Resharder
from .utils import make_data_unshard as make_data_unshard
from .utils import print_program_with_dist_attr as print_program_with_dist_attr
from .utils import set_grad_var_shape as set_grad_var_shape
from .utils import to_list as to_list

class Engine:
    model: Any = ...
    inputs_spec: Any = ...
    labels_spec: Any = ...
    cluster: Any = ...
    strategy: Any = ...
    def __init__(
        self,
        model: Any | None = ...,
        inputs_spec: Any | None = ...,
        labels_spec: Any | None = ...,
        cluster: Any | None = ...,
        strategy: Any | None = ...,
    ) -> None: ...
    mode: Any = ...
    def prepare(
        self,
        optimizer: Any | None = ...,
        loss: Any | None = ...,
        metrics: Any | None = ...,
        mode: str = ...,
        all_ranks: bool = ...,
    ) -> None: ...
    def fit(self, train_data: Any, batch_size: int = ..., epochs: int = ..., steps_per_epoch: Any | None = ...): ...
    def predict(
        self, test_data: Any, batch_size: int = ..., use_program_cache: bool = ..., return_numpy: bool = ...
    ): ...
    def save(self, path: Any, training: bool = ..., mode: Any | None = ...) -> None: ...
    def load(self, path: Any, strict: bool = ..., load_optimizer: bool = ..., mode: Any | None = ...) -> None: ...
