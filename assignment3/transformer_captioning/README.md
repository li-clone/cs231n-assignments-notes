# Transformer Captioning and Vision Transformer

状态：代码与理论问答已完成；CIFAR-10 两轮完整训练结果待在 Colab / GPU 环境确认。

对应课程 Notebook：`Transformer_Captioning.ipynb`。

已完成内容：

- Sinusoidal Positional Encoding
- Multi-Head Scaled Dot-Product Attention 与 causal mask
- Transformer Decoder Layer
- Transformer 图像描述模型
- Patch Embedding
- Transformer Encoder Layer
- Vision Transformer 前向传播
- ViT 训练超参数配置
- 3 道 Inline Question
- COCO 50 样本过拟合实验

实现验证：

```text
PatchEmbedding relative error = 5.94e-06
Captioning final loss         = 0.02237
Captioning target             = loss < 0.05
```

确定性结构、因果性和梯度测试均已通过。完整 CIFAR-10 数据下载受本地网络速度限制，`> 45%` 两轮测试准确率仍需运行 Notebook 最后部分确认。

目录内容：

- `README.md`：完成状态与结果摘要
- `notes.md`：中文学习总结

详细学习总结见 [notes.md](notes.md)。
