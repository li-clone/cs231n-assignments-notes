# Dropout

状态：已完成。

对应课程 Notebook：`Dropout.ipynb`。

已完成内容：

- Inverted Dropout 训练前向传播
- Dropout 测试模式
- Dropout 反向传播
- FullyConnectedNet 的 Dropout 支持
- Dropout 数值梯度检查
- CIFAR-10 正则化实验

实现验证：

```text
Dropout layer dx error  = 5.45e-11
Dropout network error   = 6.25e-07
Test mode identity      = True
```

最终实验结果：

```text
No Dropout best train accuracy = 99.2%
No Dropout best val accuracy   = 32.3%
Dropout best train accuracy    = 95.6%
Dropout best val accuracy      = 34.7%
```

目录内容：

- `Dropout.ipynb`：实验流程与理论问题回答
- `layers.py`：Inverted Dropout 及基础层实现
- `fc_net.py`：支持 Dropout 和归一化的全连接网络
- `notes.md`：中文学习总结

详细学习总结见 [notes.md](notes.md)。
