from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
from typing_extensions import Self

Numberic = int | float | complex | np.number[Any]

_T = TypeVar("_T", bound=Numberic)
_SeqLevel1 = tuple[_T, ...] | list[_T]

_TL1 = TypeVar("_TL1", bound=_SeqLevel1[Numberic])
_SeqLevel2 = tuple[_TL1, ...] | list[_TL1]

_TL2 = TypeVar("_TL2", bound=_SeqLevel2[_SeqLevel1[Numberic]])
_SeqLevel3 = tuple[_TL2, ...] | list[_TL2]

_TL3 = TypeVar("_TL3", bound=_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]])
_SeqLevel4 = tuple[_TL3, ...] | list[_TL3]

_TL4 = TypeVar("_TL4", bound=_SeqLevel4[_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]])
_SeqLevel5 = tuple[_TL4, ...] | list[_TL4]

_TL5 = TypeVar("_TL5", bound=_SeqLevel5[_SeqLevel4[_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]]])
_SeqLevel6 = tuple[_TL5, ...] | list[_TL5]

NumbericSequence = (
    Numberic
    | _SeqLevel1[Numberic]
    | _SeqLevel2[_SeqLevel1[Numberic]]
    | _SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]
    | _SeqLevel4[_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]]
    | _SeqLevel5[_SeqLevel4[_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]]]
    | _SeqLevel6[_SeqLevel5[_SeqLevel4[_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]]]]
)

class Tensor:
    def __sub__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
    def __rsub__(self, other: Self | np.ndarray[Any, Any] | Numberic) -> Tensor: ...
