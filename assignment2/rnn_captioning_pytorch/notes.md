# CS231n Assignment 2 — RNN Image Captioning 学习总结

## 1. 本次任务

本部分使用 PyTorch 从零实现循环神经网络，并把预提取的图片特征转换成自然语言描述。

完整实践流程：

```text
COCO Captioning Data
→ Vanilla RNN Step Forward
→ Vanilla RNN Sequence Forward
→ Word Embedding
→ Temporal Affine / Softmax
→ CaptioningRNN Training Loss
→ Small-data Overfitting
→ Autoregressive Sampling
→ LSTM Step / Sequence
```

因为使用 PyTorch，反向传播由 Autograd 完成；本作业重点是理解并实现完整前向计算图。

实际结果：

```text
Vanilla RNN final small-data loss = 0.01337
LSTM final small-data loss       = 0.09099
```

两者都达到作业要求的 `loss < 0.1`。

---

## 2. 图像描述任务

普通图像分类输出一个固定类别：

```text
image → class score
```

图像描述需要输出长度可变的单词序列：

```text
image → "a dog is running on the grass"
```

本作业不直接训练 CNN，而是使用已经从 VGG-16 提取的图片特征。PCA 将原始 4096 维特征压缩到 512 维：

```text
features.shape = (N, D)
D = 512
```

图片特征负责提供视觉信息，RNN 或 LSTM 负责根据视觉信息逐步生成语言。

---

## 3. COCO Captioning 数据

课程提供的数据包括：

- 训练和验证图片的 VGG-16 特征；
- 每张图片对应的人工 caption；
- 单词与整数 ID 的映射；
- 图片 URL；
- caption 对应的图片索引。

Caption 不直接保存为字符串，而是表示为整数序列：

```text
"a cat sits" → [token_a, token_cat, token_sits]
```

这样可以高效地组成 minibatch，并通过 Embedding 查表转换为连续向量。

---

## 4. 特殊 Token

Caption 中使用四类特殊 token：

| Token | 作用 |
|---|---|
| `<START>` | 表示生成过程开始 |
| `<END>` | 表示句子结束 |
| `<NULL>` | 把不同长度的句子补齐到相同长度 |
| `<UNK>` | 表示词表中没有收录的罕见词 |

例如：

```text
<START> a cat sleeps <END> <NULL> <NULL>
```

`<NULL>` 只是为了组成规则 Tensor，不应该参与 loss。

---

## 5. Vanilla RNN 单步前向传播

一个时间步同时接收当前输入和上一个隐藏状态：

```text
x_t       : 当前单词向量
h_(t-1)   : 前面所有信息压缩出的状态
```

公式：

\[
h_t=\tanh(x_tW_x+h_{t-1}W_h+b)
\]

代码：

```python
next_h = torch.tanh(x.mm(Wx) + prev_h.mm(Wh) + b)
```

Shape：

```text
x       : (N, D)
Wx      : (D, H)
prev_h  : (N, H)
Wh      : (H, H)
b       : (H,)
next_h  : (N, H)
```

其中：

```text
(N, D) @ (D, H) = (N, H)
(N, H) @ (H, H) = (N, H)
```

`Wh` 使用 `(H, H)`，既满足矩阵乘法，也让隐藏状态维度在不同时间步保持不变。

---

## 6. 隐藏状态就是递归记忆

完整递推过程：

```text
h0 + x0 → h1
h1 + x1 → h2
h2 + x2 → h3
...
```

当时间向前移动后，当前状态要成为下一步的上一状态：

```python
prev_h = next_h
```

如果每一步都继续使用原始 `h0`，各时间步之间不会传递信息，模型也无法根据前文预测后续单词。

---

## 7. Vanilla RNN 完整序列

输入序列：

```text
x.shape = (N, T, D)
```

- `N`：minibatch 大小；
- `T`：序列长度；
- `D`：每个时间步的输入维度。

实现：

```python
prev_h = h0
hidden_states = []

for t in range(T):
    prev_h = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
    hidden_states.append(prev_h)

h = torch.stack(hidden_states, dim=1)
```

每个隐藏状态是 `(N, H)`，沿时间维堆叠后：

```text
T × (N, H) → (N, T, H)
```

使用列表和 `torch.stack` 可以自然保留完整 Autograd 计算图。

---

## 8. 时间展开与 Autograd

RNN 在代码中复用同一组参数 `Wx`、`Wh` 和 `b`，但计算图会沿时间展开：

```text
loss_T
  ↓
h_T → h_(T-1) → ... → h_1 → h_0
```

调用：

```python
loss.backward()
```

PyTorch 会自动执行 Backpropagation Through Time，并累加共享参数在所有时间步产生的梯度。

Vanilla RNN 中反复经过 `tanh` 和 `Wh`，长序列上容易出现梯度消失或梯度爆炸。

---

## 9. Word Embedding

单词最初是整数 ID：

```text
x.shape = (N, T)
```

Embedding 表：

```text
W.shape = (V, D)
```

- `V`：词表大小；
- `D`：词向量维度；
- `W[i]`：第 `i` 个 token 的向量。

前向传播只是查表：

```python
out = W[x]
```

形状变化：

```text
(N, T) → (N, T, D)
```

相同单词具有相同 token ID，因此会先查询同一行 `W[index]`。它在不同上下文中的含义由后续 RNN/LSTM 隐藏状态进一步建模。

Autograd 会把同一 token 在不同位置产生的梯度累加到相同的 Embedding 行。

---

## 10. Temporal Affine Layer

RNN 在每个时间步输出隐藏状态：

```text
h.shape = (N, T, H)
```

需要把每个隐藏状态转换成整个词表的分数：

```python
scores = temporal_affine_forward(h, W_vocab, b_vocab)
```

参数与输出：

```text
W_vocab : (H, V)
b_vocab : (V,)
scores  : (N, T, V)
```

这相当于对全部 `N×T` 个隐藏向量共享同一个全连接层。

---

## 11. Temporal Softmax Loss 与 Mask

每个时间步都要预测一个词，因此对 `(N, T)` 的每个有效位置计算 Cross Entropy。

不同 caption 长度不同，短句会补 `<NULL>`。Mask 定义为：

```python
mask = captions_out != self._null
```

计算过程：

```python
x_flat = scores.reshape(N * T, V)
y_flat = captions_out.reshape(N * T).long()
mask_flat = mask.reshape(N * T)

loss = F.cross_entropy(x_flat, y_flat, reduction="none")
loss = loss * mask_flat.float()
loss = loss.sum() / N
```

`<NULL>` 位置乘以 0，不影响 loss 和梯度。

在 Windows 上，NumPy 的 `randint` 可能产生 `int32`；PyTorch Cross Entropy 要求标签为 Long，因此显式使用 `.long()` 能保证跨平台运行。

---

## 12. Caption 的输入与目标错位

真实 caption：

```text
<START>  a  cat  sleeps  <END>  <NULL>
```

训练时切成：

```text
captions_in:
<START>  a    cat     sleeps  <END>

captions_out:
a        cat  sleeps  <END>   <NULL>
```

代码：

```python
captions_in = captions[:, :-1]
captions_out = captions[:, 1:]
```

错开一位表示“依据当前词预测下一个词”：

```text
输入 <START> → 预测 a
输入 a       → 预测 cat
输入 cat     → 预测 sleeps
```

如果输入和目标完全相同，模型可能只学习复制当前词，而不是语言的下一个词条件分布。

---

## 13. CaptioningRNN 训练前向传播

完整训练链路：

```python
h0 = affine_forward(features, W_proj, b_proj)
word_vectors = word_embedding_forward(captions_in, W_embed)
h = rnn_forward(word_vectors, h0, Wx, Wh, b)
scores = temporal_affine_forward(h, W_vocab, b_vocab)
loss = temporal_softmax_loss(scores, captions_out, mask)
```

Shape 流程：

```text
features       (N, D)
→ h0           (N, H)

captions_in    (N, T)
→ embeddings   (N, T, W)
→ hidden       (N, T, H)
→ scores       (N, T, V)
→ loss         scalar
```

图片特征只负责初始化 `h0`，之后视觉信息通过隐藏状态影响整个生成序列。

---

## 14. Teacher Forcing

训练时知道真实 caption，因此每个时间步可以直接输入真实的前一个词：

```text
真实 <START> → 真实 a → 真实 cat → 真实 sleeps
```

这种训练方式称为 Teacher Forcing。

优点是每个时间步都能得到正确上下文，训练更稳定、更容易并行准备输入。

缺点是训练和测试存在差异：测试时必须使用模型自己的预测。一旦测试时某一步预测错误，错误可能被送入下一步并继续累积，这称为 exposure bias。

---

## 15. 为什么要做小数据过拟合

只训练 50 条 caption，并不是为了得到有泛化能力的模型，而是进行 sanity check：

```text
如果大模型能记住几十个样本
→ 前向、梯度和优化流程大概率正确

如果连几十个样本都记不住
→ 优先检查实现、梯度、学习率和数据管线
```

本次 Vanilla RNN 的 loss：

| Iteration | Loss |
|---:|---:|
| 1 | 80.0272 |
| 21 | 3.8906 |
| 41 | 0.1332 |
| 51 | 0.0576 |
| 100 | 0.01337 |

目标为 `< 0.1`，实验通过。

严重过拟合只证明模型具备学习训练样本的能力，不代表它能泛化到未见过的图片。

---

## 16. 测试时自回归采样

测试时没有真实 caption，只知道图片和 `<START>`：

```text
输入 <START>
→ 预测 a
→ 把 a 作为下一步输入
→ 预测 cat
→ 把 cat 作为下一步输入
→ ...
```

每一步：

```python
word_vector = word_embedding_forward(prev_word, W_embed)
prev_h = rnn_step_forward(word_vector, prev_h, Wx, Wh, b)
scores = affine_forward(prev_h, W_vocab, b_vocab)
next_word = scores.argmax(dim=1)

captions[:, t] = next_word
prev_word = next_word
```

初始化：

```python
prev_h = affine_forward(features, W_proj, b_proj)
prev_word = <START>
```

这里不能直接调用完整 `rnn_forward`，因为未来时间步的输入尚不存在。必须先生成当前 token，才能确定下一步输入。

---

## 17. Greedy Decoding

本作业在每个时间步选择最高分单词：

```python
next_word = scores.argmax(dim=1)
```

这称为 greedy decoding。

优点：

- 简单；
- 速度快；
- 每一步只保留一个候选。

缺点：

- 当前局部最优选择不一定组成整体概率最高的句子；
- 早期错误会影响后续全部 token。

更完整的图像描述系统常使用 beam search，同时保留若干条累计分数最高的候选序列。

---

## 18. LSTM 为什么出现

Vanilla RNN 每一步都用：

```python
tanh(x @ Wx + prev_h @ Wh + b)
```

重写隐藏状态。长序列中，梯度需要连续穿过大量 `tanh` 和矩阵乘法，容易消失或爆炸。

LSTM 增加 cell state `c`，通过门控选择保留、写入和输出信息，使长期记忆具有更直接的传递路径。

---

## 19. LSTM 的四个门

一次性计算四组激活：

```python
activation = x.mm(Wx) + prev_h.mm(Wh) + b
ai, af, ao, ag = torch.chunk(activation, 4, dim=1)
```

因为包含四组大小为 `H` 的数据：

```text
Wx : (D, 4H)
Wh : (H, 4H)
b  : (4H,)
```

四个门：

```python
i = torch.sigmoid(ai)  # input gate
f = torch.sigmoid(af)  # forget gate
o = torch.sigmoid(ao)  # output gate
g = torch.tanh(ag)     # candidate information
```

- `i`：允许写入多少候选信息；
- `f`：保留多少旧 cell state；
- `o`：允许输出多少内部记忆；
- `g`：准备写入的新候选内容。

---

## 20. LSTM 状态更新

长期记忆：

\[
c_t=f_t\odot c_{t-1}+i_t\odot g_t
\]

```python
next_c = forget_gate * prev_c + input_gate * candidate
```

对外隐藏状态：

\[
h_t=o_t\odot\tanh(c_t)
\]

```python
next_h = output_gate * torch.tanh(next_c)
```

当：

```text
forget gate ≈ 1
input gate  ≈ 0
```

则：

```text
next_c ≈ prev_c
```

旧记忆可以沿 cell state 接近线性地传递，有助于缓解长期依赖中的梯度消失。

---

## 21. LSTM 完整序列

初始 cell state 使用 0：

```python
prev_h = h0
prev_c = torch.zeros_like(h0)
hidden_states = []

for t in range(T):
    prev_h, prev_c = lstm_step_forward(
        x[:, t, :], prev_h, prev_c, Wx, Wh, b
    )
    hidden_states.append(prev_h)

h = torch.stack(hidden_states, dim=1)
```

`c` 是 LSTM 内部记忆，不需要作为最终序列输出；对外返回各时间步的 `h`。

测试采样时则必须同时保留并更新 `prev_h` 和 `prev_c`。

---

## 22. RNN 与 LSTM 实验结果

| 模型 | Small-data Final Loss | 采样结果 |
|---|---:|---|
| Vanilla RNN | 0.01337 | 训练样本 caption 与真实文本一致 |
| LSTM | 0.09099 | 训练样本 caption 与真实文本一致 |

梯度检查：

```text
LSTM step gradcheck     = True
LSTM sequence gradcheck = True
```

这次实验只用于验证实现，并不能根据最终 loss 直接判断 Vanilla RNN 比 LSTM 更好。模型初始化、优化轨迹和小数据记忆速度都会影响该数值；LSTM 的主要优势通常体现在更长序列和更复杂的长期依赖上。

---

## 23. 字符级与单词级模型

字符级模型每个时间步预测一个字符，而不是一个完整单词。

优势：

- 字符词表通常只有几十到几百项，Embedding 和输出层更小；
- 可以逐字符生成词表外的新单词，不需要全部替换成 `<UNK>`。

劣势：

- 同一句话会变成更长的序列；
- 训练和采样更慢；
- 长期依赖更难学习；
- 模型还必须学习单词拼写。

本次 Inline Question 答案：

> 优势：词表和输出层更小，而且可以生成词表外的新单词。
>
> 劣势：序列更长，训练与采样更慢，长期依赖更难学习。

---

## 24. 常见易错点

### 1. 忘记传递隐藏状态

每步必须令当前 `next_h` 成为下一步 `prev_h`。

### 2. 时间维堆叠错误

需要得到 `(N, T, H)`，因此使用：

```python
torch.stack(hidden_states, dim=1)
```

### 3. Embedding 写成矩阵乘法

输入是整数索引，应直接使用 `W[x]` 查表。

### 4. Caption 输入输出没有错位

训练目标是预测下一个词，而不是复制当前词。

### 5. `<NULL>` 参与 loss

必须用 mask 排除补齐位置。

### 6. Cross Entropy 标签 dtype 错误

标签必须转成 `torch.long`。

### 7. 测试时继续使用真实 caption

测试阶段没有答案，必须把预测词反馈到下一时间步。

### 8. LSTM 门顺序不一致

本实现的顺序是：

```text
input, forget, output, candidate
```

拆分、初始化和计算必须使用相同顺序。

### 9. LSTM 采样忘记 cell state

RNN 只维护 `h`，LSTM 必须同时维护 `h` 与 `c`。

### 10. 用小数据结果判断泛化性能

小数据过拟合只是调试信号，不是验证集或测试集指标。

---

## 25. 本次知识链路

```text
Image Feature
→ Initial Hidden State
→ Token IDs
→ Word Embedding
→ RNN / LSTM Sequence Model
→ Hidden State at Every Timestep
→ Vocabulary Scores
→ Masked Temporal Softmax
→ Teacher-forced Training
→ Autoregressive Sampling
```

这部分把 Assignment 2 前面学到的仿射变换、Softmax、PyTorch Autograd 和优化器，扩展到了长度可变的序列生成任务。

---

## 26. 复习检查

完成本部分后，应能回答：

1. 为什么图片特征可以用来初始化 RNN 的 `h0`？
2. 为什么 `Wh` 的形状是 `(H, H)`？
3. 为什么每个时间步都要执行 `prev_h = next_h`？
4. Word Embedding 为什么可以用 `W[x]` 实现？
5. `captions_in` 与 `captions_out` 为什么要错开一位？
6. `<NULL>` mask 如何影响 loss 和梯度？
7. Teacher Forcing 与测试时自回归生成有什么区别？
8. 为什么小数据严重过拟合是一个有用的调试信号？
9. LSTM 的 input、forget、output gate 分别控制什么？
10. 为什么 LSTM 比 Vanilla RNN 更容易保存长期信息？
11. Greedy decoding 有什么局限？
12. 字符级语言模型相对单词级模型有什么优缺点？
