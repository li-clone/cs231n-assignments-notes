# Convolutional Networks

状态：已完成。

对应课程 Notebook：`ConvolutionalNetworks.ipynb`。

已完成内容：

- Naive Convolution 前向与反向传播
- Max Pooling 前向与反向传播
- Cython Fast Convolution 扩展
- ThreeLayerConvNet 参数初始化与完整传播
- 小数据过拟合实验
- CIFAR-10 完整训练
- Spatial Batch Normalization
- Spatial Group Normalization
- 全部数值梯度检查

最终结果：

```text
Training Accuracy   = 45.6%
Validation Accuracy = 45.9%
```

目录内容：

- `ConvolutionalNetworks.ipynb`：完整实验流程
- `layers.py`：卷积、池化与空间归一化实现
- `cnn.py`：三层卷积网络
- `notes.md`：中文学习总结

详细学习总结见 [notes.md](notes.md)。
