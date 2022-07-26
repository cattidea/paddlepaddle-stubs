from __future__ import annotations

from typing import Any

from paddle.dataset.image import batch_images_from_tar as batch_images_from_tar
from paddle.dataset.image import load_image as load_image

from .common import download as download

DATA_URL: str
LABEL_URL: str
SETID_URL: str
DATA_MD5: str
LABEL_MD5: str
SETID_MD5: str
TRAIN_FLAG: str
TEST_FLAG: str
VALID_FLAG: str

def default_mapper(is_train: Any, sample: Any): ...

train_mapper: Any
test_mapper: Any

def reader_creator(
    data_file: Any,
    label_file: Any,
    setid_file: Any,
    dataset_name: Any,
    mapper: Any,
    buffered_size: int = ...,
    use_xmap: bool = ...,
    cycle: bool = ...,
): ...
def train(mapper: Any = ..., buffered_size: int = ..., use_xmap: bool = ..., cycle: bool = ...): ...
def test(mapper: Any = ..., buffered_size: int = ..., use_xmap: bool = ..., cycle: bool = ...): ...
def valid(mapper: Any = ..., buffered_size: int = ..., use_xmap: bool = ...): ...
def fetch() -> None: ...
