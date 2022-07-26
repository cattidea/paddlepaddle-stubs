from __future__ import annotations

from typing import Any

from .controller import ControleMode as ControleMode
from .controller import Controller as Controller

class PSController(Controller):
    @classmethod
    def enable(cls, ctx: Any): ...
    def build_pod(self) -> None: ...
