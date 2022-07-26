from __future__ import annotations

from typing import Any, Optional

def global_scope(): ...
def scope_guard(scope: Any) -> None: ...

class FetchHandler:
    var_dict: Any = ...
    period_secs: Any = ...
    def __init__(self, var_dict: Optional[Any] = ..., period_secs: int = ...) -> None: ...
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
    def __init__(self, place: Optional[Any] = ...) -> None: ...
    def close(self) -> None: ...
    def run(
        self,
        program: Optional[Any] = ...,
        feed: Optional[Any] = ...,
        fetch_list: Optional[Any] = ...,
        feed_var_name: str = ...,
        fetch_var_name: str = ...,
        scope: Optional[Any] = ...,
        return_numpy: bool = ...,
        use_program_cache: bool = ...,
        return_merged: bool = ...,
        use_prune: bool = ...,
    ): ...
    def infer_from_dataset(
        self,
        program: Optional[Any] = ...,
        dataset: Optional[Any] = ...,
        scope: Optional[Any] = ...,
        thread: int = ...,
        debug: bool = ...,
        fetch_list: Optional[Any] = ...,
        fetch_info: Optional[Any] = ...,
        print_period: int = ...,
        fetch_handler: Optional[Any] = ...,
    ): ...
    def start_heter_trainer(
        self,
        program: Optional[Any] = ...,
        scope: Optional[Any] = ...,
        debug: bool = ...,
        fetch_list: Optional[Any] = ...,
        fetch_info: Optional[Any] = ...,
        print_period: int = ...,
        fetch_handler: Optional[Any] = ...,
    ): ...
    def train_from_dataset(
        self,
        program: Optional[Any] = ...,
        dataset: Optional[Any] = ...,
        scope: Optional[Any] = ...,
        thread: int = ...,
        debug: bool = ...,
        fetch_list: Optional[Any] = ...,
        fetch_info: Optional[Any] = ...,
        print_period: int = ...,
        fetch_handler: Optional[Any] = ...,
    ): ...
