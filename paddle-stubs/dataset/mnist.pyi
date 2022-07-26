from __future__ import annotations

from typing import Any

URL_PREFIX: str
TEST_IMAGE_URL: Any
TEST_IMAGE_MD5: str
TEST_LABEL_URL: Any
TEST_LABEL_MD5: str
TRAIN_IMAGE_URL: Any
TRAIN_IMAGE_MD5: str
TRAIN_LABEL_URL: Any
TRAIN_LABEL_MD5: str

def reader_creator(image_filename: Any, label_filename: Any, buffer_size: Any): ...
def train(): ...
def test(): ...
def fetch() -> None: ...
