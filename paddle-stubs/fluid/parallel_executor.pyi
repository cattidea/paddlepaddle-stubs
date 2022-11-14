from __future__ import annotations

from typing import Any, Optional

class ParallelExecutor:
    def __init__(
        self,
        use_cuda: Any,
        loss_name: str | None = ...,
        main_program: Any | None = ...,
        share_vars_from: Any | None = ...,
        exec_strategy: Any | None = ...,
        build_strategy: Any | None = ...,
        num_trainers: int = ...,
        trainer_id: int = ...,
        scope: Any | None = ...,
    ) -> None: ...
    def run(self, fetch_list: Any, feed: Any | None = ..., feed_dict: Any | None = ..., return_numpy: bool = ...): ...
    @property
    def device_count(self): ...
    def drop_local_exe_scopes(self) -> None: ...
