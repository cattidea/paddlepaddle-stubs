from __future__ import annotations

from typing import Any, Optional

def data(name: Any, shape: Any, dtype: Any | None = ..., lod_level: int = ...): ...

class InputSpec:
    shape: Any = ...
    dtype: Any = ...
    name: Any = ...
    def __init__(self, shape: Any, dtype: str = ..., name: str | None = ...) -> None: ...
    @classmethod
    def from_tensor(cls, tensor: Any, name: str | None = ...): ...
    @classmethod
    def from_numpy(cls, ndarray: Any, name: str | None = ...): ...
    def batch(self, batch_size: Any): ...
    def unbatch(self): ...
    def __hash__(self) -> Any: ...
    def __eq__(self, other: Any) -> Any: ...
    def __ne__(self, other: Any) -> Any: ...
