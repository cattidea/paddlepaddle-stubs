from __future__ import annotations

from typing import Any

from .. import core as core
from . import collective as collective

OpRole: Any

class AscendTranspiler(collective.Collective):
    nrings: int = ...
    def __init__(self, startup_program: Any, main_program: Any) -> None: ...
    def transpile(self) -> None: ...
