from __future__ import annotations

from typing import Any, Optional

def list_to_ordered_dict(list_obj: Any, ordered_dict: Any | None = ...): ...
def get_inputs_of_program(program: Any): ...
def get_outputs_of_program(program: Any): ...
def prune_program(program: Any, start_op_idx: Any, end_op_idx: Any): ...
def split_program(program: Any, op_indices: Any): ...
