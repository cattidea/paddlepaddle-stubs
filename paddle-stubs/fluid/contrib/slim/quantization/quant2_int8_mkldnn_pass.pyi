from __future__ import annotations

from typing import Any, Optional

class Quant2Int8MkldnnPass:
    def __init__(
        self,
        _ops_to_quantize: Any,
        _op_ids_to_skip: Optional[Any] = ...,
        _scope: Optional[Any] = ...,
        _place: Optional[Any] = ...,
        _core: Optional[Any] = ...,
        _debug: bool = ...,
    ) -> None: ...
    def apply(self, graph: Any): ...
    def prepare_and_optimize_fp32(self, graph: Any): ...
