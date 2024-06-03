from __future__ import annotations

import paddle
from paddle import Tensor, nn


class MyModel(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_size: int = 10,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_channels)

    # Pylance will report a `reportIncompatibleMethodOverride` error here.
    # see also: https://github.com/microsoft/pyright/issues/1787
    def forward(self, x: Tensor):  # pyright: ignore[reportIncompatibleMethodOverride]
        x = self.fc1(x)
        x = self.fc2(x)
        return x


x = paddle.to_tensor([])
model = MyModel(2, 3, hidden_size=4)
# By default, the infered type is `Any`, since the `Layer.__call__` method is not overridden.
y: Tensor = model(x)
