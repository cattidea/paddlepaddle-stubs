from __future__ import annotations

from typing import Any, Optional

DATA_URL: str
DATA_MD5: str
WORDDICT_URL: str
WORDDICT_MD5: str
VERBDICT_URL: str
VERBDICT_MD5: str
TRGDICT_URL: str
TRGDICT_MD5: str
EMB_URL: str
EMB_MD5: str
UNK_IDX: int

def load_label_dict(filename: Any): ...
def load_dict(filename: Any): ...
def corpus_reader(data_path: Any, words_name: Any, props_name: Any): ...
def reader_creator(
    corpus_reader: Any,
    word_dict: Optional[Any] = ...,
    predicate_dict: Optional[Any] = ...,
    label_dict: Optional[Any] = ...,
): ...
def get_dict(): ...
def get_embedding(): ...
def test(): ...
def fetch() -> None: ...
