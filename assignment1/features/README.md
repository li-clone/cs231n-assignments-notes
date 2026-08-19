# Image Features

状态：已完成。

对应课程 Notebook：`features.ipynb`。

完成内容：

- 提取 HOG 与颜色直方图特征
- 特征标准化与 bias trick
- 在特征上训练 Softmax 分类器
- 在特征上训练 Two-Layer Neural Network
- 使用 validation set 调参并分析错误样本

最终结果：

```text
Softmax on Features
Validation Accuracy: 43.6%
Test Accuracy:       43.2%

Two-Layer Net on Features
Validation Accuracy: 61.8%
Test Accuracy:       60.3%
```

文件：

- `features.ipynb`：完整实验、调参与 Inline Question
- `features.py`：HOG、HSV 颜色直方图与特征提取实现
- `notes.md`：中文学习总结
