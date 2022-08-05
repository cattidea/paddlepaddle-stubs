from __future__ import annotations

import collections
from collections.abc import Sequence
from typing import Any, Optional

from typing_extensions import Literal

from ..._typing import NumbericSequence

class Compose:
    transforms: Any = ...
    def __init__(self, transforms: Any) -> None: ...
    def __call__(self, data: Any): ...

class BaseTransform:
    keys: Any = ...
    params: Any = ...
    def __init__(self, keys: Optional[Any] = ...) -> None: ...
    def __call__(self, inputs: Any): ...

class ToTensor(BaseTransform):
    data_format: Any = ...
    def __init__(self, data_format: str = ..., keys: Optional[Any] = ...) -> None: ...

class Resize(BaseTransform):
    size: Any = ...
    interpolation: Any = ...
    def __init__(self, size: Any, interpolation: str = ..., keys: Optional[Any] = ...) -> None: ...

class RandomResizedCrop(BaseTransform):
    size: Any = ...
    scale: Any = ...
    ratio: Any = ...
    interpolation: Any = ...
    def __init__(
        self, size: Any, scale: Any = ..., ratio: Any = ..., interpolation: str = ..., keys: Optional[Any] = ...
    ) -> None: ...

class CenterCrop(BaseTransform):
    size: Any = ...
    def __init__(self, size: Any, keys: Optional[Any] = ...) -> None: ...

class RandomHorizontalFlip(BaseTransform):
    prob: Any = ...
    def __init__(self, prob: float = ..., keys: Optional[Any] = ...) -> None: ...

class RandomVerticalFlip(BaseTransform):
    prob: Any = ...
    def __init__(self, prob: float = ..., keys: Optional[Any] = ...) -> None: ...

class Normalize(BaseTransform):
    mean: Any = ...
    std: Any = ...
    data_format: Any = ...
    to_rgb: Any = ...
    def __init__(
        self,
        mean: NumbericSequence = ...,
        std: NumbericSequence = ...,
        data_format: Literal["HWC", "CHW"] = ...,
        to_rgb: bool = ...,
        keys: Optional[Sequence[str]] = ...,
    ) -> None: ...

class Transpose(BaseTransform):
    order: Any = ...
    def __init__(self, order: Any = ..., keys: Optional[Any] = ...) -> None: ...

class BrightnessTransform(BaseTransform):
    value: Any = ...
    def __init__(self, value: Any, keys: Optional[Any] = ...) -> None: ...

class ContrastTransform(BaseTransform):
    value: Any = ...
    def __init__(self, value: Any, keys: Optional[Any] = ...) -> None: ...

class SaturationTransform(BaseTransform):
    value: Any = ...
    def __init__(self, value: Any, keys: Optional[Any] = ...) -> None: ...

class HueTransform(BaseTransform):
    value: Any = ...
    def __init__(self, value: Any, keys: Optional[Any] = ...) -> None: ...

class ColorJitter(BaseTransform):
    brightness: Any = ...
    contrast: Any = ...
    saturation: Any = ...
    hue: Any = ...
    def __init__(
        self,
        brightness: int = ...,
        contrast: int = ...,
        saturation: int = ...,
        hue: int = ...,
        keys: Optional[Any] = ...,
    ) -> None: ...

class RandomCrop(BaseTransform):
    size: Any = ...
    padding: Any = ...
    pad_if_needed: Any = ...
    fill: Any = ...
    padding_mode: Any = ...
    def __init__(
        self,
        size: Any,
        padding: Optional[Any] = ...,
        pad_if_needed: bool = ...,
        fill: int = ...,
        padding_mode: str = ...,
        keys: Optional[Any] = ...,
    ) -> None: ...

class Pad(BaseTransform):
    padding: Any = ...
    fill: Any = ...
    padding_mode: Any = ...
    def __init__(self, padding: Any, fill: int = ..., padding_mode: str = ..., keys: Optional[Any] = ...) -> None: ...

class RandomRotation(BaseTransform):
    degrees: Any = ...
    interpolation: Any = ...
    expand: Any = ...
    center: Any = ...
    fill: Any = ...
    def __init__(
        self,
        degrees: Any,
        interpolation: str = ...,
        expand: bool = ...,
        center: Optional[Any] = ...,
        fill: int = ...,
        keys: Optional[Any] = ...,
    ) -> None: ...

class Grayscale(BaseTransform):
    num_output_channels: Any = ...
    def __init__(self, num_output_channels: int = ..., keys: Optional[Any] = ...) -> None: ...
