# Stanford CS231n Assignments & Notes

这是我学习 Stanford CS231n 时整理的作业实现与中文学习笔记，涵盖 Assignment 1、Assignment 2 和 Assignment 3，共 14 个小作业，目前已全部完成。

## 学习方式

这些笔记主要由我与 Codex 通过逐题问答共同完成：

```text
阅读题目与代码
→ Codex 提出知识点问题
→ 我先用自己的理解回答
→ 共同检查并补充关键细节
→ 确认理解后写入代码或文字答案
→ 运行题目测试与真实实验
→ 整理为中文复习笔记
```

## 完成进度

| Assignment | 内容 | 进度 |
|---|---|---:|
| [Assignment 1](assignment1/) | 基础分类器、损失函数、反向传播与全连接网络 | 5 / 5 |
| [Assignment 2](assignment2/) | 归一化、正则化、卷积网络、PyTorch 与图像描述 | 5 / 5 |
| [Assignment 3](assignment3/) | Transformer、自监督学习、扩散模型、CLIP 与 DINO | 4 / 4 |
| **总计** | **14 个小作业** | **14 / 14** |

### Assignment 1

- [x] [k-Nearest Neighbor](assignment1/knn/)
- [x] [Softmax Classifier](assignment1/softmax/)
- [x] [Two-Layer Neural Network](assignment1/two_layer_net/)
- [x] [Image Features](assignment1/features/)
- [x] [Multi-Layer Fully Connected Networks](assignment1/fully_connected_nets/)

### Assignment 2

- [x] [Batch Normalization](assignment2/batch_normalization/)
- [x] [Dropout](assignment2/dropout/)
- [x] [Convolutional Networks](assignment2/convolutional_networks/)
- [x] [PyTorch](assignment2/pytorch/)
- [x] [RNN Image Captioning](assignment2/rnn_captioning_pytorch/)

### Assignment 3

- [x] [Transformer Captioning](assignment3/transformer_captioning/)
- [x] [Self-Supervised Learning and SimCLR](assignment3/self_supervised_learning/)
- [x] [Denoising Diffusion Probabilistic Models](assignment3/ddpm/)
- [x] [CLIP and DINO](assignment3/clip_dino/)

## 仓库涵盖的内容

主要知识点包括：

- kNN、Softmax 与线性分类器
- 全连接网络、反向传播与优化算法
- Batch Normalization、Dropout 与卷积神经网络
- PyTorch 模型训练与 RNN 图像描述
- Transformer、Self-Attention 与 Cross-Attention
- SimCLR、数据增强与对比损失
- DDPM、条件 U-Net 与 Classifier-Free Guidance
- CLIP 零样本分类、跨模态检索与共享 embedding 空间
- DINO patch features、自注意力可视化与轻量语义分割

除题目代码和 Notebook 外，仓库还保留了部分真实实验产物，例如训练指标、生成图片、特征可视化和分割视频。

## 目录结构

```text
cs231n-assignments-notes/
├── assignment1/
│   ├── knn/
│   ├── softmax/
│   ├── two_layer_net/
│   ├── features/
│   └── fully_connected_nets/
├── assignment2/
│   ├── batch_normalization/
│   ├── dropout/
│   ├── convolutional_networks/
│   ├── pytorch/
│   └── rnn_captioning_pytorch/
└── assignment3/
    ├── transformer_captioning/
    ├── self_supervised_learning/
    ├── ddpm/
    └── clip_dino/
```

每个小作业目录通常包含：

```text
README.md   完成状态、实现内容与实验摘要
notes.md    中文知识总结、易错点与复习问题
*.ipynb     已完成的课程 Notebook
*.py        本次作业涉及的核心实现
*.png/mp4/json  部分作业产生的图片、视频或指标文件
```

课程提供但没有修改的通用框架、数据集和大型预训练权重不在本仓库重复保存。

## 实验与验证

代码完成后不仅运行了 Notebook 自带测试，也对关键任务进行了实际训练或推理，包括：

- SimCLR weighted kNN 与 linear evaluation；
- 文本条件 DDPM Emoji 生成；
- CLIP zero-shot classification 与文本检索；
- DINO patch attention、PCA 及 DAVIS 视频分割。

详细实现说明、公式推导和实验数字请进入对应目录查看 `notes.md`。

## 说明

本仓库用于个人课程学习、复习与实验记录，不代表 Stanford 官方解答。代码应结合课程题目、自己的推导和测试结果阅读。
