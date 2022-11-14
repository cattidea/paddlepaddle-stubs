from __future__ import annotations

from typing import Any, Optional

class Quant2Int8MkldnnPass:
    def __init__(
        self,
        _ops_to_quantize: Any,
        _op_ids_to_skip: Any | None = ...,
        _scope: Any | None = ...,
        _place: Any | None = ...,
        _core: Any | None = ...,
        _debug: bool = ...,
    ) -> None: ...
    def apply(self, graph: Any): ...
    def prepare_and_optimize_fp32(self, graph: Any): ...
