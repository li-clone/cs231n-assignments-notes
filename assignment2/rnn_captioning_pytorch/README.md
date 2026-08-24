# RNN Image Captioning (PyTorch)

状态：已完成。

对应课程 Notebook：`RNN_Captioning_pytorch.ipynb`。

已完成内容：

- Vanilla RNN 单步与完整序列前向传播
- Word Embedding
- Temporal Affine 与 Masked Softmax
- CaptioningRNN 训练 loss
- 小数据过拟合实验
- 测试时自回归采样
- LSTM 单步与完整序列
- 字符级与单词级模型比较

最终结果：

```text
Vanilla RNN final loss = 0.01337
LSTM final loss       = 0.09099
```

目录内容：

- `RNN_Captioning_pytorch.ipynb`：完整实验流程
- `rnn_layers_pytorch.py`：RNN、Embedding、LSTM 与时间层实现
- `rnn_pytorch.py`：CaptioningRNN 训练和采样实现
- `notes.md`：中文学习总结

详细学习总结见 [notes.md](notes.md)。
