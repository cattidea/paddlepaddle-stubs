from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import paddle
from paddle.vision.transforms import Normalize

if TYPE_CHECKING:
    import numpy.typing as npt

transform = Normalize(mean=[127.5], std=[127.5], data_format="CHW")
# 下载数据集并初始化 DataSet
train_dataset = paddle.vision.datasets.MNIST(mode="train", transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode="test", transform=transform)

# 模型组网并初始化网络
lenet = paddle.vision.models.LeNet(num_classes=10)
model = paddle.Model(lenet)

# 模型训练的配置准备，准备损失函数，优化器和评价指标
model.prepare(
    paddle.optimizer.Adam(parameters=model.parameters()),
    paddle.nn.CrossEntropyLoss(),
    paddle.metric.Accuracy(),
)

# 模型训练
model.fit(train_dataset, epochs=5, batch_size=64, verbose=1)
# 模型评估
model.evaluate(test_dataset, batch_size=64, verbose=1)

# 保存模型
model.save("./output/mnist")
# 加载模型
model.load("output/mnist")

# 从测试集中取出一张图片
img, label = test_dataset[0]
# 将图片 shape 从 1*28*28 变为 1*1*28*28，增加一个 batch 维度，以匹配模型输入格式要求
img_batch: npt.NDArray[np.float32] = np.expand_dims(img.astype("float32"), axis=0)  # type: ignore

# 执行推理并打印结果，此处 predict_batch 返回的是一个 list，取出其中数据获得预测结果
out = model.predict_batch(img_batch)[0]
pred_label = out.argmax()
print("true label: {}, pred label: {}".format(label[0], pred_label))
