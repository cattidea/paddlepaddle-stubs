from __future__ import annotations

from typing_extensions import Literal

DataLayout0D = Literal["NC"]
DataLayout1D = Literal["NCL", "NLC"]
DataLayout2D = Literal["NCHW", "NHCW"]
DataLayout3D = Literal["NCDHW", "NDHWC"]
DataLayoutND = DataLayout0D | DataLayout1D | DataLayout2D | DataLayout3D

DataLayout1DVariant = Literal["NCW", "NWC"]
DataLayoutImage = Literal["HWC", "CHW"]
