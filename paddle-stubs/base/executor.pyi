from __future__ import annotations

from typing import Any

import numpy.typing as npt

def global_scope(): ...
def scope_guard(scope: Any) -> None: ...

class FetchHandler:
    var_dict: Any = ...
    period_secs: Any = ...
    def __init__(self, var_dict: Any | None = ..., period_secs: int = ...) -> None: ...
    def handler(self, res_dict: Any) -> None: ...
    @staticmethod
    def help() -> None: ...

class _StandaloneExecutor:
    def __init__(self, place: Any, main_program: Any, scope: Any) -> None: ...
    def run(self, feed_names: Any, fetch_list: Any, return_numpy: bool = ...): ...

class _ExecutorCache:
    def __init__(self, place: Any) -> None: ...

class Executor:
    place: Any = ...
    program_caches: Any = ...
    ctx_caches: Any = ...
    trainer_caches: Any = ...
    scope_caches: Any = ...
    var_caches: Any = ...
    pruned_program_caches: Any = ...
    pruned_program_scope_caches: Any = ...
    def __init__(self, place: Any | None = ...) -> None: ...
    def close(self) -> None: ...
    def run(
        self,
        program: Any | None = ...,
        feed: Any | None = ...,
        fetch_list: Any | None = ...,
        feed_var_name: str = ...,
        fetch_var_name: str = ...,
        scope: Any | None = ...,
        return_numpy: bool = ...,
        use_program_cache: bool = ...,
        return_merged: bool = ...,
        use_prune: bool = ...,
    ) -> list[npt.NDArray[Any]]: ...
    def infer_from_dataset(
        self,
        program: Any | None = ...,
        dataset: Any | None = ...,
        scope: Any | None = ...,
        thread: int = ...,
        debug: bool = ...,
        fetch_list: Any | None = ...,
        fetch_info: Any | None = ...,
        print_period: int = ...,
        fetch_handler: Any | None = ...,
    ): ...
    def start_heter_trainer(
        self,
        program: Any | None = ...,
        scope: Any | None = ...,
        debug: bool = ...,
        fetch_list: Any | None = ...,
        fetch_info: Any | None = ...,
        print_period: int = ...,
        fetch_handler: Any | None = ...,
    ): ...
    def train_from_dataset(
        self,
        program: Any | None = ...,
        dataset: Any | None = ...,
        scope: Any | None = ...,
        thread: int = ...,
        debug: bool = ...,
        fetch_list: Any | None = ...,
        fetch_info: Any | None = ...,
        print_period: int = ...,
        fetch_handler: Any | None = ...,
    ): ...
