from __future__ import annotations

import abc
from typing import Any, Optional

from ..base.data_feeder import check_variable_and_dtype as check_variable_and_dtype
from ..base.framework import core as core
from ..base.layer_helper import LayerHelper as LayerHelper

class Metric(metaclass=abc.ABCMeta):
    def __init__(self) -> None: ...
    @abc.abstractmethod
    def reset(self) -> Any: ...
    @abc.abstractmethod
    def update(self, *args: Any) -> Any: ...
    @abc.abstractmethod
    def accumulate(self) -> Any: ...
    @abc.abstractmethod
    def name(self) -> Any: ...
    def compute(self, *args: Any): ...

class Accuracy(Metric):
    topk: Any = ...
    maxk: Any = ...
    def __init__(self, topk: Any = ..., name: str | None = ..., *args: Any, **kwargs: Any) -> None: ...
    def compute(self, pred: Any, label: Any, *args: Any): ...
    def update(self, correct: Any, *args: Any): ...
    total: Any = ...
    count: Any = ...
    def reset(self) -> None: ...
    def accumulate(self): ...
    def name(self): ...

class Precision(Metric):
    tp: int = ...
    fp: int = ...
    def __init__(self, name: str = ..., *args: Any, **kwargs: Any) -> None: ...
    def update(self, preds: Any, labels: Any) -> None: ...
    def reset(self) -> None: ...
    def accumulate(self): ...
    def name(self): ...

class Recall(Metric):
    tp: int = ...
    fn: int = ...
    def __init__(self, name: str = ..., *args: Any, **kwargs: Any) -> None: ...
    def update(self, preds: Any, labels: Any) -> None: ...
    def accumulate(self): ...
    def reset(self) -> None: ...
    def name(self): ...

class Auc(Metric):
    def __init__(
        self, curve: str = ..., num_thresholds: int = ..., name: str = ..., *args: Any, **kwargs: Any
    ) -> None: ...
    def update(self, preds: Any, labels: Any) -> None: ...
    @staticmethod
    def trapezoid_area(x1: Any, x2: Any, y1: Any, y2: Any): ...
    def accumulate(self): ...
    def reset(self) -> None: ...
    def name(self): ...

def accuracy(
    input: Any,
    label: Any,
    k: int = ...,
    correct: Any | None = ...,
    total: Any | None = ...,
    name: str | None = ...,
): ...
