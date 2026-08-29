# Self-Supervised Learning

状态：已完成。

对应课程 Notebook：`Self_Supervised_Learning.ipynb`。

已完成内容：

- SimCLR 随机数据增强与正样本对生成
- 余弦相似度
- 循环版 NT-Xent Loss
- 正样本对批量相似度
- 完整相似度矩阵
- 向量化 NT-Xent Loss
- SimCLR 单轮继续训练
- Weighted kNN 表示评估
- 随机特征与预训练特征的线性评估

实验结果：

```text
SimCLR train loss              = 3.264274
Weighted kNN Top-1 / Top-5     = 83.34% / 99.25%
Random frozen features Top-1   = 16.69%
SimCLR frozen features Top-1   = 82.57%
Linear evaluation requirement  = >= 70%
```

目录内容：

- `Self_Supervised_Learning.ipynb`：完整实验流程
- `data_utils.py`：SimCLR 数据增强和正样本对生成
- `contrastive_loss.py`：循环与向量化 NT-Xent Loss
- `model.py`：ResNet-50 Encoder 和 Projection Head
- `utils.py`：SimCLR 训练、weighted kNN 与线性评估工具
- `simclr_linear_eval_results.json`：两组线性评估的逐 epoch 结果
- `README.md`：完成状态与结果摘要
- `notes.md`：中文学习总结

详细学习总结见 [notes.md](notes.md)。
