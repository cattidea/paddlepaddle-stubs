from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeVar

import numpy.typing as npt
from PIL.Image import Image as PILImage
from typing_extensions import TypeAlias

from ..._typing import DataLayoutImage, NumbericSequence, Tensor
from ..._typing.basic import IntSequence

_DataT = TypeVar("_DataT", bound=Tensor | PILImage | npt.NDArray[Any])
_InterpolationPil: TypeAlias = Literal["nearest", "bilinear", "bicubic", "lanczos", "hamming"]
_InterpolationCv2: TypeAlias = Literal["nearest", "bilinear", "area", "bicubic", "lanczos"]

class Compose:
    transforms: Any = ...
    def __init__(
        self,
        transforms: list[BaseTransform] | tuple[BaseTransform, ...],
    ) -> None: ...
    def __call__(self, data: _DataT) -> _DataT: ...

class BaseTransform:
    keys: Any = ...
    params: Any = ...
    def __init__(self, keys: Sequence[str] | None = ...) -> None: ...
    def __call__(self, inputs: _DataT) -> _DataT: ...

class ToTensor(BaseTransform):
    data_format: Any = ...
    def __init__(self, data_format: str = ..., keys: Any | None = ...) -> None: ...
    def __call__(self, inputs: _DataT) -> Tensor: ...  # pyright: ignore [reportInvalidTypeVarUse]

class Resize(BaseTransform):
    size: Any = ...
    interpolation: Any = ...
    def __init__(
        self,
        size: int | list[int] | tuple[int, ...],
        interpolation: _InterpolationPil | _InterpolationCv2 = ...,
        keys: Sequence[str] | None = ...,
    ) -> None: ...
    def __call__(self, inputs: _DataT) -> _DataT: ...

class RandomResizedCrop(BaseTransform):
    size: Any = ...
    scale: Any = ...
    ratio: Any = ...
    interpolation: Any = ...
    def __init__(
        self,
        size: int | list[int] | tuple[int, ...],
        scale: list[float] | tuple[float, ...] = ...,
        ratio: list[float] | tuple[float, ...] = ...,
        interpolation: _InterpolationPil | _InterpolationCv2 = ...,
        keys: Sequence[str] | None = ...,
    ) -> None: ...

class CenterCrop(BaseTransform):
    size: Any = ...
    def __init__(
        self,
        size: int | list[int] | tuple[int, ...],
        keys: Sequence[str] | None = ...,
    ) -> None: ...

class RandomHorizontalFlip(BaseTransform):
    prob: Any = ...
    def __init__(
        self,
        prob: float = ...,
        keys: Sequence[str] | None = ...,
    ) -> None: ...

class RandomVerticalFlip(BaseTransform):
    prob: Any = ...
    def __init__(
        self,
        prob: float = ...,
        keys: Sequence[str] | None = ...,
    ) -> None: ...

class Normalize(BaseTransform):
    mean: Any = ...
    std: Any = ...
    data_format: Any = ...
    to_rgb: Any = ...
    def __init__(
        self,
        mean: NumbericSequence = ...,
        std: NumbericSequence = ...,
        data_format: DataLayoutImage = ...,
        to_rgb: bool = ...,
        keys: Sequence[str] | None = ...,
    ) -> None: ...

class Transpose(BaseTransform):
    order: Any = ...
    def __init__(
        self,
        order: Sequence[int] = ...,
        keys: Sequence[str] | None = ...,
    ) -> None: ...

class BrightnessTransform(BaseTransform):
    value: Any = ...
    def __init__(self, value: float, keys: Sequence[str] | None = ...) -> None: ...

class ContrastTransform(BaseTransform):
    value: Any = ...
    def __init__(self, value: float, keys: Sequence[str] | None = ...) -> None: ...

class SaturationTransform(BaseTransform):
    value: Any = ...
    def __init__(self, value: float, keys: Sequence[str] | None = ...) -> None: ...

class HueTransform(BaseTransform):
    value: Any = ...
    def __init__(self, value: float, keys: Sequence[str] | None = ...) -> None: ...

class ColorJitter(BaseTransform):
    brightness: Any = ...
    contrast: Any = ...
    saturation: Any = ...
    hue: Any = ...
    def __init__(
        self,
        brightness: float = ...,
        contrast: float = ...,
        saturation: float = ...,
        hue: float = ...,
        keys: Sequence[str] | None = ...,
    ) -> None: ...

class RandomCrop(BaseTransform):
    size: Any = ...
    padding: Any = ...
    pad_if_needed: Any = ...
    fill: Any = ...
    padding_mode: Any = ...
    def __init__(
        self,
        size: int | IntSequence,
        padding: int | IntSequence | None = ...,
        pad_if_needed: bool = ...,
        fill: int = ...,
        padding_mode: str = ...,
        keys: Sequence[str] | None = ...,
    ) -> None: ...

class Pad(BaseTransform):
    padding: Any = ...
    fill: Any = ...
    padding_mode: Any = ...
    def __init__(
        self,
        padding: int | IntSequence,
        fill: int | IntSequence = ...,
        padding_mode: str = ...,
        keys: Sequence[str] | None = ...,
    ) -> None: ...

class RandomRotation(BaseTransform):
    degrees: Any = ...
    interpolation: Any = ...
    expand: Any = ...
    center: Any = ...
    fill: Any = ...
    def __init__(
        self,
        degrees: float | Sequence[float],
        interpolation: _InterpolationPil | _InterpolationCv2 = ...,
        expand: bool = ...,
        center: tuple[int, int] | None = ...,
        fill: float = ...,
        keys: Sequence[str] | None = ...,
    ) -> None: ...

class Grayscale(BaseTransform):
    num_output_channels: Any = ...
    def __init__(
        self,
        num_output_channels: Literal[1, 3] = ...,
        keys: Sequence[str] | None = ...,
    ) -> None: ...
