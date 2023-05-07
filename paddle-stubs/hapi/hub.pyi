from __future__ import annotations

from typing import Any, List, Literal  # noqa: Y022

from ..nn import Layer

DEFAULT_CACHE_DIR: str
VAR_DEPENDENCY: str
MODULE_HUBCONF: str
HUB_DIR: Any

def list(
    repo_dir: str,
    source: Literal["github", "gitee", "local"] = ...,
    force_reload: bool = ...,
) -> List[Any]: ...  # noqa: UP006 list[Any], but list has conflicts with function name
def help(
    repo_dir: str,
    model: str,
    source: Literal["github", "gitee", "local"] = ...,
    force_reload: bool = ...,
) -> str: ...
def load(
    repo_dir: str,
    model: str,
    source: Literal["github", "gitee", "local"] = ...,
    force_reload: bool = ...,
    **kwargs: Any,
) -> Layer: ...
