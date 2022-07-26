from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from typing import Any

from .. import Layer

class LayerDict(Layer):
    def __init__(
        self, sublayers: OrderedDict[str, Layer] | list[tuple[str, Layer]] | LayerDict | None = ...
    ) -> None: ...
    def __getitem__(self, key: str) -> Layer: ...
    def __setitem__(self, key: str, sublayer: Layer) -> None: ...
    def __delitem__(self, key: Any) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Any: ...
    def __contains__(self, key: Layer) -> bool: ...
    def clear(self) -> None: ...
    def pop(self, key: Any) -> Layer: ...
    def keys(self) -> Iterator[str]: ...
    def items(self) -> Iterator[tuple[str, Layer]]: ...
    def values(self) -> Iterator[Layer]: ...
    def update(self, sublayers: Any) -> None: ...
