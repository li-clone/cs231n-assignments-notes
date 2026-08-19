# Two-Layer Neural Network

状态：已完成。

实现内容：

- Affine forward / backward
- ReLU forward / backward
- Softmax loss
- 模块化 `Affine → ReLU → Affine` 网络
- L2 regularization
- Numerical gradient checking
- Solver + SGD 训练
- Loss / Accuracy 曲线与第一层权重可视化
- 隐藏层容量调参和过拟合分析

最终结果：

```text
Validation Accuracy: 52.7%
Test Accuracy:       49.6%
```

文件：

- `two_layer_net.ipynb`：完整实验过程与 Inline Questions
- `layers.py`：基础层实现
- `layer_utils.py`：组合层
- `fc_net.py`：TwoLayerNet 实现
- `notes.md`：中文学习总结
