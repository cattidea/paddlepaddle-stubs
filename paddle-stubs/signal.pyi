from __future__ import annotations

from . import Tensor

def stft(
    x: Tensor,
    n_fft: int,
    hop_length: int | None = ...,
    win_length: int | None = ...,
    window: Tensor | None = ...,
    center: bool = ...,
    pad_mode: str = ...,
    normalized: bool = ...,
    onesided: bool = ...,
    name: str | None = ...,
) -> Tensor: ...
def istft(
    x: Tensor,
    n_fft: int,
    hop_length: int | None = ...,
    win_length: int | None = ...,
    window: Tensor | None = ...,
    center: bool = ...,
    normalized: bool = ...,
    onesided: bool = ...,
    length: int | None = ...,
    return_complex: bool = ...,
    name: str | None = ...,
) -> Tensor: ...
