from __future__ import annotations

from typing import Any, Optional

def parse_args(): ...
def transform_and_save_int8_model(
    original_path: Any,
    save_path: Any,
    ops_to_quantize: str = ...,
    op_ids_to_skip: str = ...,
    debug: bool = ...,
    quant_model_filename: str = ...,
    quant_params_filename: str = ...,
    save_model_filename: str = ...,
    save_params_filename: Optional[Any] = ...,
) -> None: ...
