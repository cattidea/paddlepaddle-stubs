from __future__ import annotations

from typing import Any, Sequence, TypeVar

import numpy as np

Numberic = int | float | complex | np.number[Any]

_T = TypeVar("_T", bound=Numberic)
_SeqLevel1 = Sequence[_T]

_TL1 = TypeVar("_TL1", bound=_SeqLevel1[Numberic])
_SeqLevel2 = Sequence[_TL1]

_TL2 = TypeVar("_TL2", bound=_SeqLevel2[_SeqLevel1[Numberic]])
_SeqLevel3 = Sequence[_TL2]

_TL3 = TypeVar("_TL3", bound=_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]])
_SeqLevel4 = Sequence[_TL3]

_TL4 = TypeVar("_TL4", bound=_SeqLevel4[_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]])
_SeqLevel5 = Sequence[_TL4]

_TL5 = TypeVar("_TL5", bound=_SeqLevel5[_SeqLevel4[_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]]])
_SeqLevel6 = Sequence[_TL5]

IntSequence = _SeqLevel1[int]
NumbericSequence = _SeqLevel1[Numberic]
NestedNumbericSequence = (
    Numberic
    | _SeqLevel1[Numberic]
    | _SeqLevel2[_SeqLevel1[Numberic]]
    | _SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]
    | _SeqLevel4[_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]]
    | _SeqLevel5[_SeqLevel4[_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]]]
    | _SeqLevel6[_SeqLevel5[_SeqLevel4[_SeqLevel3[_SeqLevel2[_SeqLevel1[Numberic]]]]]]
)
