from __future__ import annotations

from typing import Any, Optional

from .layers import Layer

class Sequential(Layer):
    def __init__(self, *layers: Any) -> None: ...
    def __getitem__(self, name: Any): ...
    def __setitem__(self, name: Any, layer: Any) -> None: ...
    def __delitem__(self, name: Any) -> None: ...
    def __len__(self): ...
    def forward(self, input: Any): ...

class ParameterList(Layer):
    def __init__(self, parameters: Optional[Any] = ...) -> None: ...
    def __getitem__(self, idx: Any): ...
    def __setitem__(self, idx: Any, param: Any) -> None: ...
    def __len__(self): ...
    def __iter__(self) -> Any: ...
    def append(self, parameter: Any): ...

class LayerList(Layer):
    def __init__(self, sublayers: Optional[Any] = ...) -> None: ...
    def __getitem__(self, idx: Any): ...
    def __setitem__(self, idx: Any, sublayer: Any): ...
    def __delitem__(self, idx: Any) -> None: ...
    def __len__(self): ...
    def __iter__(self) -> Any: ...
    def append(self, sublayer: Any): ...
    def insert(self, index: Any, sublayer: Any) -> None: ...
    def extend(self, sublayers: Any): ...
