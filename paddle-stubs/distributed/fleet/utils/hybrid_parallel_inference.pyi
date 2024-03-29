from __future__ import annotations

from typing import Any, Optional

from paddle.base.framework import Operator as Operator

class HybridParallelInferenceHelper:
    ring_id: int = ...
    micro_batch_size: Any = ...
    beam_size: Any = ...
    init_comm: Any = ...
    role_maker: Any = ...
    mp_ring_id: int = ...
    global_ring_id: int = ...
    endpoints: Any = ...
    current_endpoint: Any = ...
    rank: Any = ...
    nranks: Any = ...
    num_pp: Any = ...
    num_mp: Any = ...
    global_endpoints: Any = ...
    global_rank: Any = ...
    global_nranks: Any = ...
    mp_group: Any = ...
    pp_group: Any = ...
    def __init__(
        self,
        startup_program: Any,
        main_program: Any,
        num_mp: int = ...,
        num_pp: int = ...,
        micro_batch_size: int = ...,
        beam_size: int = ...,
        init_comm: bool = ...,
        role_maker: Any | None = ...,
    ) -> None: ...
    def gen_infer_program(
        self,
        sync_in_while_lastpp2firstpp_var_names: Any | None = ...,
        sync_in_while_var_names: Any | None = ...,
        debug: bool = ...,
    ) -> None: ...
