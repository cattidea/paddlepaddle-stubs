from __future__ import annotations

from typing import Any, Optional

from .. import core as core
from .. import framework as framework
from ..distribute_lookup_table import (
    find_distributed_lookup_table as find_distributed_lookup_table,
)
from ..framework import Block as Block
from ..framework import Parameter as Parameter
from ..framework import Program as Program
from ..framework import default_main_program as default_main_program
from ..framework import default_startup_program as default_startup_program
from .details import VarsDistributed as VarsDistributed
from .details import delete_ops as delete_ops
from .details import wait_server_ready as wait_server_ready
from .distribute_transpiler import DistributeTranspiler as DistributeTranspiler
from .distribute_transpiler import (
    DistributeTranspilerConfig as DistributeTranspilerConfig,
)
from .distribute_transpiler import ServerRuntimeConfig as ServerRuntimeConfig
from .distribute_transpiler import same_or_split_var as same_or_split_var
from .distribute_transpiler import slice_variable as slice_variable
from .ps_dispatcher import PSDispatcher as PSDispatcher
from .ps_dispatcher import RoundRobin as RoundRobin

RPC_OP_ROLE_ATTR_NAME: Any
op_role_attr_name: Any
RPC_OP_ROLE_ATTR_VALUE: Any

class GeoSgdTranspiler(DistributeTranspiler):
    config: Any = ...
    def __init__(self, config: Any | None = ...) -> None: ...
    origin_program: Any = ...
    startup_program: Any = ...
    origin_startup_program: Any = ...
    trainer_num: Any = ...
    sync_mode: bool = ...
    trainer_id: Any = ...
    pserver_endpoints: Any = ...
    vars_overview: Any = ...
    param_name_to_grad_name: Any = ...
    grad_name_to_param_name: Any = ...
    table_name: Any = ...
    has_distributed_lookup_table: Any = ...
    vars_info: Any = ...
    split_to_origin_mapping: Any = ...
    delta_vars_list: Any = ...
    sparse_var_list: Any = ...
    sparse_var_splited_list: Any = ...
    sparse_var: Any = ...
    sparse_tables: Any = ...
    trainer_startup_program: Any = ...
    def transpile(
        self,
        trainer_id: Any,
        program: Any | None = ...,
        pservers: str = ...,
        trainers: int = ...,
        sync_mode: bool = ...,
        startup_program: Any | None = ...,
        current_endpoint: str = ...,
    ) -> None: ...
    def get_trainer_program(self, wait_port: bool = ...): ...
    param_grad_ep_mapping: Any = ...
    def get_pserver_programs(self, endpoint: Any): ...
    pserver_program: Any = ...
    def get_pserver_program(self, endpoint: Any): ...