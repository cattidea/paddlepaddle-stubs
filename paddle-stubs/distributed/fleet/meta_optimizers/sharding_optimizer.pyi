from __future__ import annotations

from typing import Any, Optional

from .common import OP_ROLE_KEY as OP_ROLE_KEY
from .common import OP_ROLE_VAR_KEY as OP_ROLE_VAR_KEY
from .common import CollectiveHelper as CollectiveHelper
from .common import OpRole as OpRole
from .common import is_backward_op as is_backward_op
from .common import is_optimizer_op as is_optimizer_op
from .common import is_update_op as is_update_op
from .meta_optimizer_base import MetaOptimizerBase as MetaOptimizerBase
from .sharding import utils as utils
from .sharding.fp16_helper import FP16Utils as FP16Utils
from .sharding.gradient_clip_helper import GradientClipHelper as GradientClipHelper
from .sharding.offload_helper import OffloadHelper as OffloadHelper
from .sharding.prune import ProgramDeps as ProgramDeps
from .sharding.shard import ProgramSegment as ProgramSegment
from .sharding.shard import Shard as Shard
from .sharding.utils import *
from .sharding.weight_decay_helper import WeightDecayHelper as WeightDecayHelper

logger: Any
formatter: Any
ch: Any

class ShardingOptimizer(MetaOptimizerBase):
    inner_opt: Any = ...
    meta_optimizers_white_list: Any = ...
    meta_optimizers_black_list: Any = ...
    mp_degree: int = ...
    def __init__(self, optimizer: Any) -> None: ...
    def minimize_impl(
        self,
        loss: Any,
        startup_program: Any | None = ...,
        parameter_list: Any | None = ...,
        no_grad_set: Any | None = ...,
    ): ...
    def collect_segment(self, segment: Any, op_idx: Any, block: Any): ...
    def create_persistable_gradients_and_insert_merge_ops(
        self, main_block: Any, startup_block: Any, insert_idx: Any, grad_names: Any, shard: Any
    ) -> None: ...
