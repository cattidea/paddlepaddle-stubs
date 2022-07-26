from __future__ import annotations

from typing import Any, Optional

class ParallelExecutor:
    def __init__(
        self,
        use_cuda: Any,
        loss_name: Optional[Any] = ...,
        main_program: Optional[Any] = ...,
        share_vars_from: Optional[Any] = ...,
        exec_strategy: Optional[Any] = ...,
        build_strategy: Optional[Any] = ...,
        num_trainers: int = ...,
        trainer_id: int = ...,
        scope: Optional[Any] = ...,
    ) -> None: ...
    def run(
        self, fetch_list: Any, feed: Optional[Any] = ..., feed_dict: Optional[Any] = ..., return_numpy: bool = ...
    ): ...
    @property
    def device_count(self): ...
    def drop_local_exe_scopes(self) -> None: ...
