# CS231n Assignment 3 — Transformer Captioning 与 ViT 学习总结

## 1. 本次任务

本部分从零实现 Transformer 的核心组件，并将同一套序列建模方法用于两个视觉任务：

```text
图像描述：Image Features → Transformer Decoder → Caption
图像分类：Image Patches  → Transformer Encoder → Class Scores
```

完整实践流程：

```text
Scaled Dot-Product Attention
→ Multi-Head Attention
→ Causal Mask
→ Positional Encoding
→ Transformer Decoder Layer
→ Image Captioning
→ Patch Embedding
→ Transformer Encoder Layer
→ Vision Transformer
```

实际验证结果：

```text
PatchEmbedding relative error = 5.94e-06
Captioning small-data loss     = 0.02237
```

Captioning 达到作业要求的 `loss < 0.05`。ViT 的完整 CIFAR-10 两轮训练仍需在数据可用时运行并确认 `test accuracy > 0.45`。

---

## 2. Transformer 处理 Token 序列

Transformer 并不限定 token 必须是文字。只要把输入转换成一组固定维度的向量，就可以使用 Attention 建模它们之间的关系。

```text
word IDs → word embeddings → text tokens
image → non-overlapping patches → patch embeddings → visual tokens
```

因此，图像描述和 ViT 虽然输出目标不同，但共享 Multi-Head Attention、位置编码、残差连接、LayerNorm 和 Feed-Forward Network。

---

## 3. Query、Key 和 Value

Attention 可以理解为一次可学习的信息检索：

- Query：当前需要寻找什么信息；
- Key：每条候选信息用于匹配的标签；
- Value：匹配后真正取回的内容。

\[
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
\]

Query 与 Key 的点积表示匹配程度，softmax 将匹配分数变成权重，再对 Value 做加权求和：

\[
\operatorname{Attention}(Q,K,V)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_{head}}}\right)V
\]

Self-Attention 中，`Q`、`K`、`V` 来自同一序列；Cross-Attention 中，Query 与 Key/Value 来自不同来源。

---

## 4. 为什么要缩放 Attention

每个注意力头的维度为：

\[
d_{head}=\frac{d}{h}
\]

Query 和 Key 的点积是 `d_head` 项乘积之和，其方差会随维度增长。不缩放时，较大的 logits 会使 softmax 很快接近 one-hot：

```text
softmax([20, 1, -10]) ≈ [1, 0, 0]
```

softmax 饱和后梯度很小，训练不稳定。除以 `sqrt(d_head)` 可以稳定点积尺度，使 softmax 分布和梯度处于合理范围。

重点不是单纯防止数值溢出，而是：

> 防止 softmax 过早饱和，并保持稳定梯度。

---

## 5. Multi-Head Attention

单个注意力头只产生一种 Attention 分布，可能把多种关系混合到一个加权平均中。多头注意力让不同头学习不同关系，例如主体、动作、空间关系和背景。

Shape 变化：

```text
Q, K, V: (N, S, E)
       → (N, S, H, E/H)
       → (N, H, S, E/H)

scores : (N, H, S, T)
output : (N, H, S, E/H)
       → (N, S, E)
```

核心实现：

```python
scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
weights = F.softmax(scores, dim=-1)
output = torch.matmul(weights, v)
output = output.transpose(1, 2).contiguous().reshape(N, S, E)
output = projection(output)
```

最后的线性投影学习如何混合不同头的信息，并将拼接结果映射回共享特征空间。没有它，各头会停留在固定、分离的子空间中。

---

## 6. Causal Mask

图像描述是自回归任务。预测第 `t` 个位置时，只能使用当前位置及之前的 token，不能读取未来的正确单词。

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

不允许关注的位置在 softmax 前被替换成负无穷：

```python
scores = scores.masked_fill(mask == 0, float("-inf"))
```

softmax 后，这些位置的权重变成 0。

需要区分：mask 限制的是同一个样本内部 token 之间的信息流；batch 中不同样本本来就是独立计算的，不会通过 Attention 互相传递信息。

---

## 7. Positional Encoding

Attention 本身只根据 token 内容计算关系，不知道序列顺序。本作业使用正弦位置编码：

\[
P_{i,2k}=\sin\left(i\cdot10000^{-2k/d}\right)
\]

\[
P_{i,2k+1}=\cos\left(i\cdot10000^{-2k/d}\right)
\]

最终输入为：

\[
X_{input}=X+P
\]

不同维度使用不同频率，使每个位置获得不同模式，并帮助模型学习相对位置关系。

位置编码注册为 buffer：

```python
self.register_buffer("pe", pe)
```

它不是可学习参数，但会随模型保存，也会在 `.to(device)` 时一起移动。

---

## 8. Transformer Decoder Layer

图像描述 Decoder 包含三个子层：

```text
Masked Self-Attention
→ Cross-Attention
→ Feed-Forward Network
```

### Self-Attention

关注字幕中已经出现的词。训练时是正确 caption 的前缀，推理时是模型已经生成的词。

### Cross-Attention

使用字幕特征作为 Query，图像特征作为 Key 和 Value：

\[
Q=\text{caption features},\qquad K,V=\text{image memory}
\]

例如已经生成 `a dog is`，Cross-Attention 会结合图片判断下一个词更可能是 `running` 还是 `sleeping`。

### Feed-Forward Network

对每个 token 独立应用相同的 MLP：

```text
Linear → GELU → Dropout → Linear
```

Attention 负责 token 之间的信息交换，FFN 负责每个 token 内部的非线性特征变换。

每个子层都使用：

```text
Sub-layer → Dropout → Residual Add → LayerNorm
```

残差连接提供直接的信息和梯度路径，LayerNorm 稳定每个 token 的特征尺度，Dropout 用于减少过拟合。

---

## 9. Transformer 图像描述模型

输入：

```text
features : (N, D)    预提取图像特征
captions : (N, T)    caption token IDs
```

完整 Shape 流程：

```text
captions (N, T)
→ Word Embedding (N, T, W)
→ Positional Encoding (N, T, W)
→ Masked Self-Attention

image features (N, D)
→ Linear Projection (N, W)
→ unsqueeze (N, 1, W)
→ Cross-Attention Memory

Decoder Output (N, T, W)
→ Vocabulary Scores (N, T, V)
```

核心实现：

```python
tgt = positional_encoding(embedding(captions))
memory = visual_projection(features).unsqueeze(1)
tgt_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
decoded = transformer(tgt, memory, tgt_mask=tgt_mask)
scores = output_layer(decoded)
```

训练时可以并行输入完整 `captions_in`，因为 causal mask 防止每个位置读取未来答案。生成时未来 token 不存在，只能从 `<START>` 开始逐词预测。

---

## 10. 小数据过拟合实验

使用 50 条 COCO caption 训练 100 个 epoch，是一次实现正确性检查：

```text
如果模型能记住少量数据
→ 前向传播、mask、梯度和优化流程大概率正确

如果连少量数据都无法拟合
→ 优先检查实现，而不是增加模型规模
```

本次结果：

```text
iterations = 200
final loss = 0.02237
target     = loss < 0.05
```

该结果只证明模型具备学习训练样本的能力，不代表它可以泛化到未见图片。

---

## 11. Patch Embedding

ViT 先把二维图片转换成 patch token。输入为：

```text
x.shape = (N, C, H, W)
```

使用边长为 `P` 的 patch：

\[
n=\frac{H}{P}\frac{W}{P},\qquad patch\_dim=CP^2
\]

例如 RGB `32×32` 图片使用 `8×8` patch：

```text
num_patches = (32 / 8)^2 = 16
patch_dim    = 3 × 8 × 8 = 192

(N, 3, 32, 32)
→ (N, 16, 192)
→ Linear Projection
→ (N, 16, D)
```

无循环切分：

```python
patches = x.reshape(N, C, H // P, P, W // P, P)
patches = patches.permute(0, 2, 4, 1, 3, 5)
patches = patches.reshape(N, num_patches, patch_dim)
tokens = projection(patches)
```

`permute` 的目标是让 patch 网格位置先排列，再把同一 patch 内的通道和像素连续展平。

---

## 12. Transformer Encoder 与 ViT

Encoder Layer：

```text
Self-Attention → Feed-Forward
```

Decoder Layer：

```text
Masked Self-Attention → Cross-Attention → Feed-Forward
```

ViT 做整图分类时，所有 patch 从一开始都已知，不存在未来信息泄露，因此不需要 causal mask。同一张图片的各 patch 可以互相关注。

特别注意：

> ViT 的 Attention 发生在同一张图片内部的 patch 之间，不是让不同 batch 的图片互相交流。

本作业的 ViT 前向流程：

```text
Image
→ Patch Embedding
→ Positional Encoding
→ Transformer Encoder
→ Mean Pooling over Patch Tokens
→ Linear Classification Head
```

Shape：

```text
image   : (N, C, H, W)
patches : (N, n, D)
encoded : (N, n, D)
pooled  : (N, D)
logits  : (N, num_classes)
```

原始 ViT 常使用额外的 `[CLS]` token；本作业对全部 patch token 做平均池化，让所有区域共同参与分类。

---

## 13. ViT 为什么更依赖数据

CNN 自带适合图像的归纳偏置：

- 局部连接；
- 卷积核权重共享；
- 平移等变性；
- 分层扩大感受野。

Vanilla ViT 的图像专用先验更弱，需要从数据中自己学会局部结构和空间规律，因此在小数据集上通常不如 CNN。

改善方法：

- 大规模监督或自监督预训练；
- 知识蒸馏，例如 DeiT；
- Mixup、CutMix、RandAugment 等强数据增强；
- weight decay、dropout、stochastic depth；
- convolutional stem 或 local attention；
- 根据数据规模控制模型容量。

---

## 14. ViT 自注意力复杂度

设 `L` 为层数，`n` 为 patch 数，`d` 为隐藏维度，`P` 为 patch 边长。忽略 QKV 和输出投影，自注意力复杂度为：

\[
O(Ln^2d),\qquad n=\frac{HW}{P^2}
\]

| 改动 | Token 数变化 | 计算量变化 |
|---|---:|---:|
| 隐藏维度 `d` 翻倍 | 不变 | `2×` |
| 高度和宽度都翻倍 | `4n` | `16×` |
| patch 边长 `P` 翻倍 | `n/4` | `1/16×` |
| 层数 `L` 翻倍 | 不变 | `2×` |

图像高宽都翻倍时，面积和 token 数变成 4 倍。Attention 对 token 数是平方复杂度，因此：

\[
(4n)^2=16n^2
\]

---

## 15. ViT 超参数

### Patch Size

较小 patch 保留更多局部细节，但 token 更多，Attention 计算显著增加。较大 patch 更快，但可能丢失细粒度信息。

### Hidden Dimension 与层数

更大的隐藏维和更多层提高模型容量，也增加计算量和小数据过拟合风险。

### Number of Heads

总隐藏维固定时，增加 head 数会减小每个 head 的维度。更多头不一定始终更好，每个头维度过小也会限制表达能力。

本次配置：

```python
learning_rate = 1e-3
weight_decay = 1e-4
batch_size = 128

model = VisionTransformer(
    patch_size=4,
    embed_dim=128,
    num_layers=4,
    num_heads=4,
    dim_feedforward=256,
    dropout=0.1,
)
```

该配置仍需通过完整两轮训练确认最终准确率。

---

## 16. 常见易错点

1. **误以为 ViT 会让不同 batch 样本互相关注**：Attention 只在每个样本内部计算，batch 只是并行维度。
2. **Mask 方向写反**：字幕生成需要下三角 mask，允许当前位置读取自己和之前的位置。
3. **Softmax 维度错误**：应沿 Key 序列维度执行 `softmax(dim=-1)`。
4. **缩放维度错误**：应除以 `sqrt(head_dim)`，不是 `sqrt(embed_dim)`。
5. **拆分多头后转置错误**：计算前通常需要 `(N, H, S, E/H)`。
6. **拼接前缺少 `contiguous`**：`transpose` 后应使用 `contiguous().reshape(...)`。
7. **Captioning 没有 causal mask**：模型会读取未来 token，破坏任务定义。
8. **Cross-Attention 来源混淆**：字幕是 Query，图像 memory 是 Key 和 Value。
9. **Patch 排列顺序错误**：应保证同一 patch 的像素连续，并按空间顺序排列 patch。
10. **平均池化维度错误**：分类前对 patch 维 `dim=1` 平均，不能对 batch 维平均。
11. **把小数据过拟合当作泛化结果**：它只验证实现和优化链路。
12. **固定种子仍出现参考值差异**：不同 PyTorch 版本的 Dropout 随机掩码可能不同，应关闭 Dropout 做确定性结构、因果性和梯度测试，不能修改正确公式迎合随机序列。

---

## 17. 本次知识链路

```text
Token Representation
→ Q / K / V Projection
→ Multi-Head Scaled Dot-Product Attention
→ Masking
→ Residual + LayerNorm
→ Feed-Forward Network

Image Feature + Caption Prefix
→ Transformer Decoder
→ Next-token Scores
→ Autoregressive Caption

Image
→ Patch Tokens
→ Transformer Encoder
→ Mean Pooling
→ Image Class
```

关键收获是：文字和图片都可以表示为 token 序列，而 Attention 提供了统一的全局关系建模方法。

---

## 18. 复习检查

完成本部分后，应能回答：

1. Query、Key 和 Value 分别表示什么？
2. 为什么 Attention logits 要除以 `sqrt(d/h)`？
3. 多头 Attention 相比单头有什么优势？
4. 多头输出拼接后为什么需要线性投影？
5. 字幕生成为什么需要 causal mask？
6. ViT 为什么不需要 causal mask？
7. ViT 的 patch 是否会与 batch 中其他图片的 patch 互相关注？
8. Self-Attention 和 Cross-Attention 的 Q/K/V 分别来自哪里？
9. 为什么 Transformer 需要位置编码？
10. 训练时为什么可以并行处理完整 caption？
11. 推理时为什么必须逐词生成？
12. 图片如何被转换为 patch token 序列？
13. Transformer Encoder 和 Decoder 有什么区别？
14. ViT 为什么在小数据集上更难训练？
15. 图像高和宽都翻倍时，为什么 Attention 计算量变成 16 倍？
16. patch 边长翻倍时，为什么 Attention 计算量变成原来的 `1/16`？
17. 小数据过拟合实验能够验证什么，不能验证什么？
