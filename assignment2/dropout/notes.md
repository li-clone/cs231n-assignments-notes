# CS231n Assignment 2 — Dropout 学习总结

## 1. 本次任务

本部分实现 Inverted Dropout，并将其接入任意深度全连接网络，最后通过 CIFAR-10 小数据实验观察 Dropout 的正则化效果。

完整流程为：

```text
随机 mask
→ Inverted Dropout 前向传播
→ 测试模式
→ Dropout 反向传播
→ 数值梯度检查
→ 接入 FullyConnectedNet
→ 正则化实验
```

---

## 2. Dropout 的基本思想

训练时，Dropout 随机把部分激活设为 0，使网络不能长期依赖少数固定神经元，从而减少神经元之间的过度共同适应。

可以粗略理解为：

```text
每次迭代训练一个不同的随机子网络
+ 所有子网络共享同一套参数
+ 测试时使用完整网络
```

Dropout 是正则化方法，不是用来提高模型训练集拟合能力的方法。

---

## 3. 保留概率 `p`

本作业中的 `p` 是保留概率，不是丢弃概率：

```text
p = 1.00 → 不使用 Dropout
p = 0.75 → 平均保留 75% 的激活
p = 0.25 → 平均保留 25% 的激活
```

`dropout_keep_ratio` 越小，随机丢弃越强，正则化越强。但如果过小，也可能造成欠拟合或优化困难。

---

## 4. Inverted Dropout 前向传播

训练阶段：

```python
mask = (np.random.rand(*x.shape) < p) / p
out = x * mask
```

mask 中：

- 被丢弃的位置为 0；
- 被保留的位置为 `1/p`。

除以 `p` 能保持输出期望不变：

\[
\mathbb{E}[out]
=p\left(\frac{x}{p}\right)+(1-p)0=x
\]

如果不除以 `p`，则：

\[
\mathbb{E}[out]=px
\]

训练时激活整体变小，测试时却恢复为 `x`，训练与测试的尺度会不一致。

测试阶段直接使用：

```python
out = x
```

这就是 Inverted Dropout 的优点：缩放在训练时完成，测试阶段不需要额外操作。

---

## 5. Dropout 反向传播

反向传播必须使用前向传播时的同一个 mask：

```python
dx = dout * mask
```

前向时被丢弃的神经元没有参与当前损失，因此它的梯度也为 0。被保留位置的梯度继续带有 `1/p` 缩放。

测试阶段是恒等映射：

```python
dx = dout
```

不能在反向传播时重新采样 mask，否则前向和反向对应的就不是同一个计算图。

---

## 6. 接入 FullyConnectedNet

Dropout 放在每个隐藏层的 ReLU 后面：

```text
Affine → [Normalization] → ReLU → Dropout
```

反向传播顺序相反：

```text
Dropout → ReLU → [Normalization] → Affine
```

最终输出层保持：

```text
最后一个隐藏层 → Affine → Softmax
```

输出层不使用 Dropout。每个隐藏层都必须保存自己的 dropout cache，因为每层具有不同的前向 mask。

---

## 7. Train Mode 与 Test Mode

网络根据是否传入标签决定模式：

```text
y is not None → train mode
y is None     → test mode
```

训练模式随机丢弃激活，测试模式使用完整网络。如果测试时仍使用 Dropout：

- 同一个输入每次预测可能不同；
- 有效网络容量被无谓降低；
- 输出与训练时设计的期望不匹配。

---

## 8. 随机种子与梯度检查

数值梯度检查会多次调用前向传播：

\[
\frac{\partial L}{\partial x}
\approx\frac{L(x+h)-L(x-h)}{2h}
\]

如果两次调用使用不同 mask，损失差异会混入随机噪声，数值梯度无法与解析梯度比较。

因此梯度检查时传入固定 `seed`，使 mask 可复现。正常训练时不应固定相同 mask，否则每一步都会训练同一个子网络，失去 Dropout 的意义。

测试结果：

```text
Dropout layer dx error = 5.45e-11
Network p=0.75 error   = 6.25e-07
Network p=0.50 error   = 4.25e-08
Test mode identity     = True
```

---

## 9. 保留比例与期望测试

使用全 1 输入测试得到：

| `p` | 实际非零比例 | 输出均值 |
|---:|---:|---:|
| 0.25 | 0.2499 | 0.9995 |
| 0.50 | 0.4996 | 0.9992 |
| 0.75 | 0.7506 | 1.0008 |

实际非零比例接近 `p`，输出均值仍接近输入均值 1，说明 `1/p` 缩放正确保持了期望。

---

## 10. CIFAR-10 正则化实验

在 500 个训练样本上比较无 Dropout 与 `keep_ratio=0.25`：

| Keep ratio | 最佳训练准确率 | 最佳验证准确率 |
|---:|---:|---:|
| 1.00，无 Dropout | 99.2% | 32.3% |
| 0.25 | 95.6% | 34.7% |

使用 Dropout 后：

- 训练准确率略低，说明训练任务变难；
- 验证准确率略高；
- 训练与验证准确率的差距缩小。

这是有效正则化器的典型表现：牺牲部分训练集拟合能力，换取更好的泛化能力。

---

## 11. 常见错误

- 把 `p` 当成丢弃概率。
- mask 只有 0 和 1，忘记除以 `p`。
- 测试阶段继续随机丢弃神经元。
- 反向传播重新采样 mask，而不是复用前向 mask。
- 把 Dropout 放在最终类别分数后。
- 所有隐藏层共用一个 cache，导致反向传播使用错误的 mask。
- 正常训练时固定 seed，使每一步使用完全相同的子网络。

---

## 12. 核心记忆

```text
训练：随机丢弃 + 除以 p
测试：恒等映射
反向：乘同一个 mask
位置：隐藏层 ReLU 之后
作用：降低过拟合，提高泛化能力
```
