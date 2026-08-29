# CS231n Assignment 3 — Self-Supervised Learning 与 SimCLR 学习总结

## 1. 本次任务

本部分实现 SimCLR，通过对比学习在不使用人工类别标签的情况下学习图像表示，再用 weighted kNN 和线性分类器检验表示质量。

完整流程：

```text
Unlabeled Image
→ Two Independent Augmentations
→ Shared ResNet-50 Encoder f
→ Projection Head g
→ NT-Xent Contrastive Loss
→ Self-Supervised Representation
→ Weighted kNN / Linear Evaluation
```

实际结果：

```text
SimCLR one-epoch train loss    = 3.264274
Weighted kNN Top-1             = 83.34%
Weighted kNN Top-5             = 99.25%
Random frozen features Top-1   = 16.69%
SimCLR frozen features Top-1   = 82.57%
Required pretrained Top-1      = >= 70%
```

---

## 2. 什么是自监督学习

监督学习依赖人工标签：

```text
image + label → classifier
```

自监督学习从数据自身构造监督信号：

```text
unlabeled images
→ automatically constructed learning target
→ reusable visual representation
```

SimCLR 不直接要求模型预测 `cat` 或 `car`，而是要求模型判断两个增强视图是否来自同一张原图。模型为了完成这个任务，需要学习能够跨越裁剪、颜色和翻转变化的稳定语义特征。

---

## 3. SimCLR 的正样本与负样本

对原始图片 `x` 独立采样两次随机增强：

\[
\tilde{x}_i=t(x),\qquad \tilde{x}_j=t'(x)
\]

它们来自同一张图片，因此构成 positive pair。

一个 batch 有 `N` 张原图，每张产生两个视图，所以共有 `2N` 个增强样本：

```text
left  = [z_1, ..., z_N]
right = [z_(N+1), ..., z_(2N)]
```

本作业的排列方式中：

```text
left[k] ↔ right[k]
out[k]  ↔ out[k + N]
```

构成正样本对。对于某个 anchor，除它自己和对应正样本外的其他增强样本是负样本。

---

## 4. 为什么数据增强是任务定义的一部分

SimCLR 使用的数据增强：

```python
transforms.RandomResizedCrop(32)
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomApply([color_jitter], p=0.8)
transforms.RandomGrayscale(p=0.2)
transforms.ToTensor()
transforms.Normalize(mean, std)
```

增强不只是普通正则化，还定义了模型应该保持哪些不变性：

- 随机裁剪：同一物体的局部视图仍应具有相关语义；
- 水平翻转：左右方向变化通常不改变类别；
- Color Jitter：颜色、亮度、对比度变化不应破坏主要语义；
- 灰度化：模型不能只依赖颜色完成匹配。

增强如果太弱，模型可能利用低层像素线索找到正样本；增强如果太强，两个视图可能失去共同语义，形成错误的正样本约束。

---

## 5. 为什么必须独立增强两次

正确实现：

```python
x_i = self.transform(img)
x_j = self.transform(img)
```

每次调用都会重新采样裁剪、翻转和颜色参数，因此 `x_i` 与 `x_j` 通常不同。

错误实现：

```python
x_i = self.transform(img)
x_j = x_i
```

如果直接复制，模型只需匹配完全相同的像素，无法学习对视觉变化稳定的高级表示。

---

## 6. Encoder 与 Projection Head

模型由两部分组成：

\[
h=f(\tilde{x}),\qquad z=g(h)
\]

- `f`：ResNet-50 Encoder，输出视觉表示 `h`；
- `g`：两层 MLP Projection Head，把 `h` 映射到对比空间 `z`。

本作业模型返回：

```python
return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
```

即归一化后的 `h` 和 `z`。

对比损失在 `z` 上优化。预训练结束后丢弃 `g`，使用 `h` 做分类等下游任务。Projection Head 可以承受对比目标中特有的信息压缩，让 Encoder 的表示保留更多适合迁移的内容。

---

## 7. 余弦相似度

两个向量的余弦相似度：

\[
\operatorname{sim}(z_i,z_j)=
\frac{z_i^\top z_j}{\lVert z_i\rVert\lVert z_j\rVert}
\]

实现：

```python
similarity = torch.dot(z_i, z_j) / (
    torch.linalg.norm(z_i) * torch.linalg.norm(z_j)
)
```

普通点积同时受到方向和向量长度影响，模型可能只通过增大向量模长提高分数。余弦相似度消除模长影响，使目标主要比较方向，即表示所编码的语义是否接近。

归一化向量后：

\[
\lVert z_i\rVert=\lVert z_j\rVert=1
\]

余弦相似度就等于普通点积。

---

## 8. NT-Xent 对比损失

对 anchor `i` 和它的正样本 `j`：

\[
l(i,j)=-\log
\frac{
\exp(\operatorname{sim}(z_i,z_j)/\tau)
}{
\sum_{k=1}^{2N}\mathbb{1}_{k\ne i}
\exp(\operatorname{sim}(z_i,z_k)/\tau)
}
\]

分子是 anchor 与正样本的匹配分数。分母包含：

- 正样本；
- 所有负样本；
- 不包含 anchor 自己。

因此它可以看成一个 `2N-1` 类分类问题：给定 anchor，从其余所有增强样本中识别正确的另一个视图。

正样本必须包含在分母中，因为 softmax 概率的分母要包含所有候选类别，其中也包括正确类别。

---

## 9. 为什么排除相似度矩阵对角线

一个向量与自身的余弦相似度为 1，通常是整行最大的值。如果把自身放入分母：

```text
anchor → itself
```

会成为一个无意义且非常强的竞争项，使正样本概率下降，但模型无法通过学习改变“自己与自己完全相似”这个事实。

向量化 mask：

```python
mask = ~torch.eye(2 * N, dtype=torch.bool, device=out.device)
exponential = exponential.masked_select(mask).view(2 * N, 2 * N - 1)
```

mask 只去掉对角线，不去掉正样本位置。

---

## 10. 为什么计算两个方向

相似度本身满足：

\[
\operatorname{sim}(z_i,z_j)=\operatorname{sim}(z_j,z_i)
\]

但损失不只由分子决定。`l(i,j)` 的分母以 `i` 为 anchor，`l(j,i)` 的分母以 `j` 为 anchor，两者面对的相似度分布不同。

因此一个正样本对需要贡献两个方向：

\[
L=\frac{1}{2N}\sum_{k=1}^{N}
\left[l(k,k+N)+l(k+N,k)\right]
\]

这样 `2N` 个增强视图都会恰好作为一次 anchor。

---

## 11. Temperature 参数

相似度在指数运算前除以温度 `tau`：

\[
\exp(\operatorname{sim}/\tau)
\]

- 较小 `tau`：放大相似度差异，softmax 更尖锐，更关注难以区分的负样本；
- 较大 `tau`：缩小相似度差异，softmax 更平滑。

温度过小可能导致训练不稳定，温度过大则可能让不同样本之间的差异不够明显。本次实验使用：

```text
tau = 0.5
```

---

## 12. 循环版 Loss

循环版清楚表达了公式：

```text
for each positive pair (k, k+N):
    compute l(k, k+N)
    compute l(k+N, k)
average over 2N anchors
```

对每个 anchor，都要遍历其余 `2N-1` 个样本，因此 Python 循环开销大，难以充分利用 GPU。

循环版的价值主要是作为容易理解和调试的参考实现，之后用它验证向量化版本。

---

## 13. 完整相似度矩阵

把所有表示连接：

```python
out = torch.cat([out_left, out_right], dim=0)  # (2N, D)
```

先逐行归一化，再进行矩阵乘法：

```python
normalized_out = out / torch.linalg.norm(out, dim=1, keepdim=True)
sim_matrix = normalized_out @ normalized_out.T
```

得到：

```text
sim_matrix.shape = (2N, 2N)
sim_matrix[i, j] = cosine_similarity(out[i], out[j])
```

矩阵特点：

- 对称；
- 对角线接近 1；
- 第 `k` 行的正样本位于第 `k+N` 列；
- 第 `k+N` 行的正样本位于第 `k` 列。

---

## 14. 向量化 Loss

向量化步骤：

```python
sim_matrix = compute_sim_matrix(out)
exponential = torch.exp(sim_matrix / tau)

mask = ~torch.eye(2 * N, dtype=torch.bool, device=out.device)
denom = exponential.masked_select(mask).view(2 * N, -1).sum(dim=1, keepdim=True)

positive_sim = sim_positive_pairs(out_left, out_right)
positive_sim = torch.cat([positive_sim, positive_sim], dim=0)
numerator = torch.exp(positive_sim / tau)

loss = torch.mean(-torch.log(numerator / denom))
```

这里把 `N` 个正样本相似度复制一次，是因为前 `N` 行和后 `N` 行分别对应正样本对的两个方向。

验证结果：

```text
sim error                         <= 3.81e-08
naive loss error                  <= 5.66e-08
vectorized loss error             <= 5.66e-08
positive-pair error               <= 3.81e-08
similarity-matrix test             passed
naive/vectorized gradient test     passed
```

向量化版本不仅 loss 相同，梯度也与循环版一致。

---

## 15. SimCLR 训练循环

每个 batch 返回：

```text
x_i, x_j, target
```

自监督训练不使用 `target`。两个视图通过同一个模型，共享 Encoder 和 Projection Head 参数：

```python
_, out_left = model(x_i)
_, out_right = model(x_j)
loss = simclr_loss_vectorized(
    out_left, out_right, temperature, device=device
)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

虽然进行了两次前向传播，但它们属于同一计算图，反向传播时两个分支产生的梯度都会累加到同一组共享参数。

真实实验使用：

```text
training images = 25,000
batch size      = 64
epochs          = 1 additional epoch
learning rate   = 1e-3
weight decay    = 1e-6
temperature     = 0.5
train loss      = 3.264274
```

该 checkpoint 在课程提供的长时间预训练权重基础上继续训练一轮，不是从随机初始化只训练一轮。

---

## 16. Weighted kNN 评估

Weighted kNN 不训练新的深层分类器，而是：

1. 用 Encoder 为全部训练图片建立 feature bank；
2. 计算测试特征与 feature bank 的余弦相似度；
3. 取最相似的 `k=200` 个训练样本；
4. 使用相似度经过温度缩放后的权重进行类别投票。

结果：

```text
Top-1 = 83.34%
Top-5 = 99.25%
```

高 kNN 准确率说明：即使不训练复杂分类头，同类图片在表示空间中也已经自然聚集。

---

## 17. Linear Evaluation

Linear Evaluation 冻结 Encoder，只训练一个线性分类层：

```text
Frozen Encoder f
→ 2048-dimensional feature
→ Linear layer
→ 10 CIFAR classes
```

线性分类器能力有限，因此高准确率必须主要来自 Encoder 已经学到的可分表示。

实验使用 10% CIFAR-10 训练集，即 5,000 张带标签图片，训练 10 个 epoch。

结果：

| 特征 | 最佳测试 Top-1 |
|---|---:|
| 随机冻结 Encoder | 16.69% |
| SimCLR 冻结 Encoder | 82.57% |

SimCLR 特征在第 1 个 epoch 已达到 `77.42%`，第 9 个 epoch 达到最佳 `82.57%`，明显超过题目要求的 `70%`。

注意：Notebook 的说明文字提到 baseline 训练所有权重，但提供的代码实际执行：

```python
for param in model.f.parameters():
    param.requires_grad = False
```

因此本次严格按代码运行，比较的是“随机冻结特征”和“SimCLR 冻结特征”的线性可分性。

---

## 18. 为什么测试准确率可能高于训练准确率

本次预训练模型的训练准确率约为 `78.5%`，测试准确率达到 `82.42%`。这并不表示测试集更容易被模型记住，主要因为：

- 训练集使用随机裁剪、颜色扰动和灰度化等强增强；
- 测试集只进行 Tensor 转换与标准化；
- 训练准确率是在更困难、不断变化的增强视图上统计的。

因此训练和测试输入分布的难度不同，不能直接按普通无增强实验解释两条准确率曲线。

---

## 19. Batch Size 与负样本数量

一个 batch 有 `N` 张原图和 `2N` 个增强视图。每个 anchor 的候选集合有 `2N-1` 个元素。

batch 越大：

- 同时提供的负样本越多；
- 对比任务更丰富；
- 通常更容易学到有区分度的特征；
- 显存和计算开销也更大。

这也是经典 SimCLR 对大 batch 较敏感的原因之一。后续方法会使用 memory bank、queue、momentum encoder 或不依赖显式负样本的目标缓解这一问题。

---

## 20. 常见易错点

1. **两个视图直接复制**：必须对同一原图独立调用 transform 两次。
2. **把不同图片当作正样本**：本作业中正样本索引是 `k ↔ k+N`。
3. **使用普通点积却不归一化**：模型可能依赖向量模长提高分数。
4. **分母包含 anchor 自身**：必须排除相似度矩阵对角线。
5. **分母排除了正样本**：正样本是 softmax 的正确类别，必须保留。
6. **只计算一个方向**：每个正样本对需要 `l(i,j)` 和 `l(j,i)`。
7. **最终除以 `N`**：共有 `2N` 个 anchor，应除以 `2N`。
8. **正样本相似度复制顺序错误**：应与拼接后的前 `N`、后 `N` 行顺序一致。
9. **mask 建在错误设备上**：应使用 `out.device`，避免 CPU/GPU device mismatch。
10. **对 `h` 而不是 `z` 计算预训练损失**：SimCLR loss 应作用于 Projection Head 输出。
11. **下游分类继续使用 Projection Head**：线性评估通常使用 Encoder 表示 `h`。
12. **认为 self-supervised 完全不需要标签**：预训练不需要标签，但线性评估仍使用少量标签衡量表示质量。
13. **只比较 loss，不检查梯度**：循环版与向量化版还应验证梯度等价。

---

## 21. 本次知识链路

```text
Original Image
→ Independent Random Views
→ Positive Pair
→ Shared Encoder f
→ Representations h
→ Projection Head g
→ Normalized Embeddings z
→ Pairwise Cosine Similarity
→ NT-Xent Loss
→ Self-Supervised Pretraining
→ Discard g and Freeze f
→ Weighted kNN / Linear Classifier
```

核心结论：

> 模型不需要类别标签，也可以通过学习“同一图片的不同视图应该接近”获得具有强类别可分性的视觉表示。

---

## 22. 复习检查

完成本部分后，应能回答：

1. 自监督学习的监督信号从哪里来？
2. 为什么同一图片需要独立增强两次？
3. 哪些增强定义了 SimCLR 希望学习的不变性？
4. 一个含 `N` 张原图的 batch 为什么有 `2N` 个增强样本？
5. 本作业中第 `k` 个样本的正样本索引是什么？
6. 余弦相似度相对普通点积有什么优势？
7. NT-Xent 分母包含哪些样本，排除哪个样本？
8. 为什么正样本也必须位于分母中？
9. 为什么要同时计算 `l(i,j)` 与 `l(j,i)`？
10. 温度变小时，softmax 分布如何变化？
11. `2N × 2N` 相似度矩阵的对角线表示什么？
12. 如何不用循环计算全部两两余弦相似度？
13. Encoder `f` 与 Projection Head `g` 分别负责什么？
14. 为什么预训练完成后通常丢弃 Projection Head？
15. Weighted kNN 如何评估表示质量？
16. Linear Evaluation 为什么要冻结 Encoder？
17. 为什么 SimCLR 对 batch size 较敏感？
18. 为什么本次测试准确率可能高于训练准确率？
19. 随机冻结特征和 SimCLR 冻结特征的准确率差距说明了什么？
