# Batch Normalization and Layer Normalization

状态：已完成。

对应课程 Notebook：`BatchNormalization.ipynb`。

已完成内容：

- Batch Normalization 训练与测试前向传播
- Batch Normalization 普通与简化反向传播
- Running mean 与 running variance
- FullyConnectedNet 的 BatchNorm 支持
- 权重初始化尺度与 batch size 实验
- Layer Normalization 前向与反向传播
- FullyConnectedNet 的 LayerNorm 支持
- 数值梯度检查

实现验证：

```text
BatchNorm dx error       = 1.70e-09
BatchNorm dgamma error   = 7.42e-13
BatchNorm dbeta error    = 2.88e-12
LayerNorm dx error       = 2.11e-09
LayerNorm network error  = 4.10e-07
```

目录内容：

- `BatchNormalization.ipynb`：实验流程与理论问题回答
- `layers.py`：BatchNorm、LayerNorm 及基础层实现
- `fc_net.py`：支持 BatchNorm、LayerNorm 和 Dropout 的全连接网络
- `notes.md`：中文学习总结

详细学习总结见 [notes.md](notes.md)。
