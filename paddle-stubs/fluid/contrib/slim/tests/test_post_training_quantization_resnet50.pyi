from __future__ import annotations

from test_post_training_quantization_mobilenetv1 import TestPostTrainingQuantization

class TestPostTrainingForResnet50(TestPostTrainingQuantization):
    def test_post_training_resnet50(self) -> None: ...

class TestPostTrainingForResnet50ONNXFormat(TestPostTrainingQuantization):
    def test_post_training_resnet50(self) -> None: ...
