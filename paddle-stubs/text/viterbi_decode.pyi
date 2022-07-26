from __future__ import annotations

from typing import Any, Optional

from ..nn import Layer

def viterbi_decode(
    potentials: Any, transition_params: Any, lengths: Any, include_bos_eos_tag: bool = ..., name: Optional[Any] = ...
): ...

class ViterbiDecoder(Layer):
    transitions: Any = ...
    include_bos_eos_tag: Any = ...
    name: Any = ...
    def __init__(self, transitions: Any, include_bos_eos_tag: bool = ..., name: Optional[Any] = ...) -> None: ...
    def forward(self, potentials: Any, lengths: Any): ...
