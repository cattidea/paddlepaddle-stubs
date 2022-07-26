from __future__ import annotations

from typing import Any

logger: Any

def init_metric(
    metric_ptr: Any,
    metric_yaml_path: Any,
    cmatch_rank_var: str = ...,
    mask_var: str = ...,
    uid_var: str = ...,
    phase: int = ...,
    cmatch_rank_group: str = ...,
    ignore_rank: bool = ...,
    bucket_size: int = ...,
) -> None: ...
def print_metric(metric_ptr: Any, name: Any): ...
def print_auc(metric_ptr: Any, is_day: Any, phase: str = ...): ...
