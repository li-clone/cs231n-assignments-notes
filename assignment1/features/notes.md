# CS231n Assignment 1 — Image Features 学习总结

## 1. 这个作业的核心目的

这部分没有要求实现新的分类器，而是研究：

> 输入数据的特征表示，会怎样影响分类效果？

前面的 Softmax 和 Two-Layer Network 都直接使用原始像素：

```text
图片
→ 展平为 3072 维像素向量
→ 分类器
```

本次改为：

```text
图片
→ 提取 HOG 和 HSV 颜色直方图
→ 拼接并标准化特征
→ Softmax / Two-Layer Network
```

分类器基本不变，只改变输入表示，就获得了明显更好的准确率。

---

## 2. 原始像素的局限

CIFAR-10 图片大小为：

```text
32 × 32 × 3 = 3072
```

直接把像素展平后，每个维度只表示某个固定位置的颜色值。

这种表示存在几个问题：

- 物体轻微平移后，大量像素位置都会变化；
- 光照变化会同时改变很多像素值；
- 背景容易影响分类；
- 像素距离和真实语义相似度并不一致；
- 线性分类器很难从像素中自动组合出边缘、形状和纹理。

因此需要把原始像素转换成更适合分类的特征。

---

## 3. HOG 特征

HOG 全称为 Histogram of Oriented Gradients，即方向梯度直方图。

它关注的是局部区域中：

```text
亮度变化的方向
→ 边缘方向
→ 局部轮廓和纹理
```

大致过程：

```text
计算图像梯度
→ 得到每个位置的梯度方向和强度
→ 将图像划分为局部区域
→ 统计各方向的梯度直方图
→ 拼接成特征向量
```

HOG 的优点：

- 能描述边缘和形状；
- 对整体亮度变化比原始像素更稳定；
- 适合识别具有明显轮廓的物体。

HOG 的局限：

- 基本忽略颜色；
- 仍然是人工设计的固定规则；
- 无法理解物体的高级语义。

---

## 4. HSV 颜色直方图

RGB 将颜色表示为红、绿、蓝三个通道，而 HSV 将颜色分为：

```text
H：Hue，色相
S：Saturation，饱和度
V：Value，明度
```

本次使用 Hue 通道构建颜色直方图：

```python
color_histogram_hsv(img, nbin=25)
```

颜色直方图统计图片中各种色相所占的比例。

它的优点：

- 能描述图片整体颜色分布；
- 对物体在图片中的具体位置不敏感；
- 可以帮助区分天空、草地、水面等不同颜色环境。

它的局限：

- 基本不保留空间结构；
- 两张布局完全不同的图片可能拥有相似的颜色直方图；
- 不能单独表示物体形状。

---

## 5. 特征互补

最终特征由两部分拼接：

```python
feature_fns = [
    hog_feature,
    lambda img: color_histogram_hsv(img, nbin=25),
]
```

两种特征提供互补信息：

```text
HOG
→ 边缘、方向、形状、纹理

HSV Histogram
→ 颜色分布
```

例如：

- HOG 可以发现车辆和动物的轮廓差异；
- HSV 可以区分蓝色天空、绿色草地和水面；
- 两者一起使用比单独使用任何一种更完整。

这体现了传统计算机视觉的重要思想：

> 人工选择能够保留任务相关信息、抑制无关变化的特征。

---

## 6. Feature Matrix

`extract_features` 对每张图片依次调用所有特征函数，然后拼接结果：

```text
第 i 张图片
→ HOG vector
+ HSV histogram vector
→ X_feats[i]
```

最终矩阵：

```text
X_train_feats.shape = (49000, feature_dim)
X_val_feats.shape   = (1000, feature_dim)
X_test_feats.shape  = (1000, feature_dim)
```

每一行表示一张图片，每一列表示一个人工特征维度。

---

## 7. 特征预处理

### 减去训练集均值

```python
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat
```

每个特征维度被中心化到均值约为 0。

验证集和测试集必须使用训练集的均值，不能分别计算自己的均值，否则会泄漏数据分布信息。

### 除以训练集标准差

```python
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat
```

这样不同特征维度具有相近尺度，避免数值较大的特征主导梯度。

### 添加 Bias Dimension

训练 Softmax 时添加一列 1：

```python
X_train_feats = np.hstack([
    X_train_feats,
    np.ones((X_train_feats.shape[0], 1)),
])
```

这样偏置可以合并进 Softmax 的权重矩阵。

特征 Shape：

```text
加入 bias 后：170
删除 bias 后：169
```

TwoLayerNet 本身已经有 `b1` 和 `b2`，所以训练神经网络前必须删除人工 bias 列，并且该单元只能运行一次。

---

## 8. 在图像特征上训练 Softmax

Softmax 的算法没有变化：

```text
features
→ scores = XW
→ Softmax probability
→ Cross Entropy Loss
→ SGD 更新 W
```

变化的只是 `X`：

```text
以前：原始像素
现在：HOG + HSV 特征
```

使用 validation set 搜索：

```python
learning_rates = [5e-8, 1e-7, 2e-7]
regularization_strengths = [1e5, 2.5e5, 5e5, 1e6]
```

每组超参数都重新创建并训练一个 Softmax：

```text
新建模型
→ 训练 1500 iterations
→ 计算 train accuracy
→ 计算 validation accuracy
→ 保存结果
→ 根据 validation accuracy 更新 best_softmax
```

最佳组合：

```text
learning_rate = 5e-8
reg = 2.5e5
```

最终结果：

```text
Training Accuracy   = 42.5%
Validation Accuracy = 43.6%
Test Accuracy       = 43.2%
```

作业要求测试准确率至少达到 42%，本次结果已达标。

---

## 9. 超参数过大导致数值发散

最初尝试较大的组合时出现：

```text
divide by zero encountered in log
overflow encountered in multiply
invalid value encountered in dot
```

原因不是 Softmax 主体公式错误，而是某些超参数组合导致训练发散：

```text
过大的 learning rate / regularization update
→ W 迅速变得极大
→ scores 变得极端
→ 正确类别概率下溢为 0
→ -log(0) 得到 inf
→ W² 溢出
→ 出现 NaN
```

处理方法是围绕稳定且表现较好的区域缩小搜索范围，而不是隐藏 warning 或继续使用已经发散的模型。

---

## 10. 错误样本分析

作业将模型错误预测为每个类别的图片可视化。

观察到的误分类总体合理：

- 一些动物在轮廓和纹理上相似；
- 飞机、船和车辆可能拥有相似的水平边缘；
- 草地或绿色背景会使图片更容易被预测为 frog 或 deer；
- 蓝色背景可能让不同物体与 plane 或 ship 混淆；
- CIFAR-10 图片很小，细节有限。

这些错误说明 HOG 和 HSV 主要理解：

```text
边缘
颜色
局部纹理
```

它们并不能真正理解“图片中的物体是什么”。

---

## 11. 在图像特征上训练 Two-Layer Network

网络输入从 3072 维原始像素变成 169 维人工特征：

```text
169 维 HOG + HSV 特征
→ Affine
→ 500 个隐藏神经元
→ ReLU
→ Affine
→ 10 类 scores
→ Softmax Loss
```

第一组保守配置：

```text
weight_scale  = 1e-3（默认）
learning_rate = 1e-2
epochs        = 15
```

结果：

```text
Best Validation Accuracy = 51.8%
Test Accuracy            = 49.7%
```

训练准确率和验证准确率都偏低，而且差距不大，说明主要问题是优化不足或欠拟合，不是过拟合。

---

## 12. 调优 Two-Layer Network

标准化后的特征尺度较稳定，因此可以使用比原始像素训练更大的学习率。

调优配置：

```text
hidden_dim    = 500
weight_scale  = 1e-2
reg           = 1e-3
learning_rate = 1e-1
lr_decay      = 0.95
epochs        = 20
batch_size    = 100
optimizer     = SGD
```

调优后的最佳结果：

```text
Best Validation Accuracy = 61.8%
Test Accuracy            = 60.3%
```

相比保守配置，准确率得到大幅提升，说明原配置的学习率和初始化尺度过小，模型没有充分优化。

---

## 13. 训练曲线与过拟合

调优模型在第 14 个 epoch 左右达到：

```text
Training Accuracy   = 73.2%
Validation Accuracy = 61.8%
```

后续训练中：

```text
Training Accuracy 继续上升到约 77.6%
Validation Accuracy 回落到约 59.9%
```

这表示模型开始过拟合训练集。

`Solver` 会保存 validation accuracy 最高时的参数，并在训练结束后将这些参数恢复到模型中。因此最终 `best_net` 使用的是第 14 个 epoch 附近的参数，而不是最后一个 epoch 的参数。

这相当于一种基于验证集表现的模型选择过程。

---

## 14. 四组实验结果对比

| 模型 | 输入表示 | Test Accuracy |
|---|---|---:|
| Softmax | 原始像素 | 约 35.9% |
| Softmax | HOG + HSV | 43.2% |
| Two-Layer Network | 原始像素 | 49.6% |
| Two-Layer Network | HOG + HSV | 60.3% |

可以得到两个结论。

### 更好的特征可以提升同一个分类器

```text
Softmax：35.9% → 43.2%
Two-Layer Net：49.6% → 60.3%
```

### 更强的分类器可以更好地利用同一组特征

```text
HOG + HSV + Softmax：43.2%
HOG + HSV + Two-Layer Net：60.3%
```

最终性能同时取决于：

```text
特征表示
+ 模型容量
+ 优化和超参数
```

---

## 15. 传统计算机视觉与深度学习

传统计算机视觉流程：

```text
人工设计特征
→ HOG / SIFT / Color Histogram
→ 分类器
```

这种方式依赖研究者提前决定什么信息重要。

深度学习和 CNN 的流程：

```text
原始图片
→ 多层卷积网络自动学习特征
→ 分类器
```

CNN 的优势是可以从数据中端到端学习：

```text
低层：边缘和颜色
中层：纹理和局部形状
高层：物体部件和语义
```

本次 HOG + HSV 实验是理解 CNN 的重要过渡：先证明特征表示非常重要，再进入“让网络自己学习特征”。

---

## 16. 本次作业真正掌握的主线

```text
原始 CIFAR-10 图片
→ 提取 HOG 和 HSV 特征
→ 使用训练集统计量标准化
→ 在特征上训练 Softmax
→ 使用 validation set 搜索超参数
→ 分析错误样本
→ 删除 Softmax 使用的 bias 特征列
→ 在特征上训练 Two-Layer Network
→ 调整学习率、初始化尺度和正则化
→ 根据 validation accuracy 选择最佳模型
→ 在 test set 上进行一次最终评价
```

最重要的认识是：

> 模型看到的不是“客观世界本身”，而是我们提供给它的数据表示。一个更合适的特征空间，可以让简单模型获得显著更好的效果。

本次最终结果：

```text
Softmax on Features Test Accuracy:       43.2%
Two-Layer Net on Features Test Accuracy: 60.3%
```
