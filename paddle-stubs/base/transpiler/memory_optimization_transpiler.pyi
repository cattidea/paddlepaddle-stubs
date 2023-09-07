from __future__ import annotations

from typing import Any, Optional

def memory_optimize(
    input_program: Any,
    skip_opt_set: Any | None = ...,
    print_log: bool = ...,
    level: int = ...,
    skip_grads: bool = ...,
) -> None: ...
def release_memory(input_program: Any, skip_opt_set: Any | None = ...) -> None: ...
