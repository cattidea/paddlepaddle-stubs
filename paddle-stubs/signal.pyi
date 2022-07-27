from __future__ import annotations

from typing import Optional

from . import Tensor

def stft(
    x: Tensor,
    n_fft: int,
    hop_length: Optional[int] = ...,
    win_length: Optional[int] = ...,
    window: Optional[Tensor] = ...,
    center: bool = ...,
    pad_mode: str = ...,
    normalized: bool = ...,
    onesided: bool = ...,
    name: Optional[str] = ...,
) -> Tensor: ...
def istft(
    x: Tensor,
    n_fft: int,
    hop_length: Optional[int] = ...,
    win_length: Optional[int] = ...,
    window: Optional[Tensor] = ...,
    center: bool = ...,
    normalized: bool = ...,
    onesided: bool = ...,
    length: Optional[int] = ...,
    return_complex: bool = ...,
    name: Optional[str] = ...,
) -> Tensor: ...
