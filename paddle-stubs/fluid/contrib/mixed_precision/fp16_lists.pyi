from __future__ import annotations

from typing import Any, Optional

class AutoMixedPrecisionLists:
    white_list: Any = ...
    black_list: Any = ...
    gray_list: Any = ...
    unsupported_list: Any = ...
    black_varnames: Any = ...
    def __init__(
        self,
        custom_white_list: Any | None = ...,
        custom_black_list: Any | None = ...,
        custom_black_varnames: Any | None = ...,
    ) -> None: ...

CustomOpLists = AutoMixedPrecisionLists
