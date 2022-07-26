from __future__ import annotations

from typing import Any

from paddle.fluid.dygraph.io import INFER_PARAMS_INFO_SUFFIX as INFER_PARAMS_INFO_SUFFIX

def save(obj: Any, path: Any, protocol: int = ..., **configs: Any) -> None: ...
def load(path: Any, **configs: Any): ...
