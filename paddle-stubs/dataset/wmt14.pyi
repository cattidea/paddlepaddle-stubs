from __future__ import annotations

from typing import Any

URL_DEV_TEST: str
MD5_DEV_TEST: str
URL_TRAIN: str
MD5_TRAIN: str
URL_MODEL: str
MD5_MODEL: str
START: str
END: str
UNK: str
UNK_IDX: int

def reader_creator(tar_file: Any, file_name: Any, dict_size: Any): ...
def train(dict_size: Any): ...
def test(dict_size: Any): ...
def gen(dict_size: Any): ...
def get_dict(dict_size: Any, reverse: bool = ...): ...
def fetch() -> None: ...
