from __future__ import annotations

from typing import Any

LOOKUP_TABLE_TYPE: str

def find_distributed_lookup_table_inputs(program: Any, table_name: Any): ...
def find_distributed_lookup_table_outputs(program: Any, table_name: Any): ...
def find_distributed_lookup_table(program: Any): ...
