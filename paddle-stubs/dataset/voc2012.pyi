from __future__ import annotations

from typing import Any

VOC_URL: str
VOC_MD5: str
SET_FILE: str
DATA_FILE: str
LABEL_FILE: str
CACHE_DIR: str

def reader_creator(filename: Any, sub_name: Any): ...
def train(): ...
def test(): ...
def val(): ...
