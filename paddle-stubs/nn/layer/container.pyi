from __future__ import annotations

from typing import Any, Optional

from .. import Layer as Layer

class LayerDict(Layer):
    def __init__(self, sublayers: Optional[Any] = ...) -> None: ...
    def __getitem__(self, key: Any): ...
    def __setitem__(self, key: Any, sublayer: Any): ...
    def __delitem__(self, key: Any) -> None: ...
    def __len__(self): ...
    def __iter__(self) -> Any: ...
    def __contains__(self, key: Any): ...
    def clear(self) -> None: ...
    def pop(self, key: Any): ...
    def keys(self): ...
    def items(self): ...
    def values(self): ...
    def update(self, sublayers: Any) -> None: ...
