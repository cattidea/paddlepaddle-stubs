from __future__ import annotations

from typing import Any

class EntryAttr:
    def __init__(self) -> None: ...

class ProbabilityEntry(EntryAttr):
    def __init__(self, probability: Any) -> None: ...

class CountFilterEntry(EntryAttr):
    def __init__(self, count_filter: Any) -> None: ...
