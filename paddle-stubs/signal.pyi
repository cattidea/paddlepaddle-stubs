from __future__ import annotations

from typing import Any, Optional

def stft(
    x: Any,
    n_fft: Any,
    hop_length: Optional[Any] = ...,
    win_length: Optional[Any] = ...,
    window: Optional[Any] = ...,
    center: bool = ...,
    pad_mode: str = ...,
    normalized: bool = ...,
    onesided: bool = ...,
    name: Optional[Any] = ...,
): ...
def istft(
    x: Any,
    n_fft: Any,
    hop_length: Optional[Any] = ...,
    win_length: Optional[Any] = ...,
    window: Optional[Any] = ...,
    center: bool = ...,
    normalized: bool = ...,
    onesided: bool = ...,
    length: Optional[Any] = ...,
    return_complex: bool = ...,
    name: Optional[Any] = ...,
): ...
