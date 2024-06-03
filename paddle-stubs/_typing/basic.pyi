from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

import numpy as np
from typing_extensions import TypeAlias

Numberic: TypeAlias = int | float | complex | np.number[Any]

_T = TypeVar("_T")
NestedSequence = _T | Sequence["NestedSequence[_T]"]
IntSequence = Sequence[int]
NumbericSequence = Sequence[Numberic]
NestedNumbericSequence: TypeAlias = NestedSequence[Numberic]
