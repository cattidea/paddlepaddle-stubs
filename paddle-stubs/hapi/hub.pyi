from __future__ import annotations

from typing import Any

DEFAULT_CACHE_DIR: str
VAR_DEPENDENCY: str
MODULE_HUBCONF: str
HUB_DIR: Any

def list(repo_dir: Any, source: str = ..., force_reload: bool = ...): ...
def help(repo_dir: Any, model: Any, source: str = ..., force_reload: bool = ...): ...
def load(repo_dir: Any, model: Any, source: str = ..., force_reload: bool = ..., **kwargs: Any): ...
