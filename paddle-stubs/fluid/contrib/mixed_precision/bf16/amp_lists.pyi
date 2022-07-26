from __future__ import annotations

from typing import Any, Optional

class AutoMixedPrecisionListsBF16:
    bf16_list: Any = ...
    fp32_list: Any = ...
    gray_list: Any = ...
    bf16_initializer_list: Any = ...
    unsupported_list: Any = ...
    fp32_varnames: Any = ...
    def __init__(
        self,
        custom_bf16_list: Any | None = ...,
        custom_fp32_list: Any | None = ...,
        custom_fp32_varnames: Any | None = ...,
    ) -> None: ...
