# CS231n Assignment 2 — BatchNorm 与 LayerNorm 学习总结

## 1. 本次任务

本部分为全连接网络加入 Batch Normalization 和 Layer Normalization，并研究归一化对深层网络优化、权重初始化及 batch size 的影响。

完整实践流程为：

```text
BatchNorm 训练前向传播
→ running statistics
→ 测试前向传播
→ 普通与简化反向传播
→ 接入 FullyConnectedNet
→ 初始化尺度实验
→ batch size 实验
→ LayerNorm 前向与反向传播
→ 完整网络数值梯度检查
```

---

## 2. 为什么需要归一化

深层网络中，每一层参数持续变化，后续层接收到的激活分布也会随之变化。归一化层把激活维持在较稳定的尺度，使优化更容易，并降低网络对权重初始化的敏感性。

归一化不会把网络永久限制为零均值、单位方差。可学习参数 `gamma` 和 `beta` 可以恢复适合任务的尺度与偏移：

\[
out=\gamma\hat{x}+\beta
\]

其中 `gamma` 初始化为 1，`beta` 初始化为 0，使归一化层刚加入网络时不额外改变标准化结果。

---

## 3. BatchNorm 训练前向传播

输入形状为：

```text
x.shape = (N, D)
```

`N` 是 batch size，`D` 是特征数。BatchNorm 对每个特征跨样本统计，因此聚合轴为：

```python
axis=0
```

计算公式：

\[
\mu=\frac{1}{N}\sum_i x_i
\]

\[
\sigma^2=\frac{1}{N}\sum_i(x_i-\mu)^2
\]

\[
\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
\]

\[
out=\gamma\hat{x}+\beta
\]

本作业使用总体方差，分母为 `N`，不是 `N-1`。`eps` 必须放在平方根内部，用来防止除零并提高数值稳定性。

---

## 4. Running Statistics 与测试模式

训练时使用当前 minibatch 的均值和方差，同时更新指数移动平均：

\[
running\_mean\leftarrow m\,running\_mean+(1-m)\mu
\]

\[
running\_var\leftarrow m\,running\_var+(1-m)\sigma^2
\]

`momentum` 越接近 1，旧统计量所占比例越高，running statistics 变化越平滑。

测试时使用 `running_mean` 和 `running_var`，而不是当前测试 batch 的统计量。原因是：

- 测试时可能只有一个样本；
- 小测试 batch 的统计量不可靠；
- 一个样本的预测不应该依赖同一 batch 中还有哪些其他样本。

因此 BatchNorm 的训练和测试行为不同。

---

## 5. BatchNorm 反向传播

由：

\[
out=\gamma\hat{x}+\beta
\]

可以直接得到：

\[
d\beta=\sum_i dout_i
\]

\[
d\gamma=\sum_i dout_i\hat{x}_i
\]

\[
d\hat{x}=dout\cdot\gamma
\]

普通反向传播继续沿归一化、方差和均值等支路传播。计算图发生分叉时，各条路径传回的梯度必须相加。

化简后输入梯度为：

\[
dx=\frac{1}{N\sqrt{\sigma^2+\epsilon}}
\left[N d\hat{x}-\sum_i d\hat{x}_i-
\hat{x}\sum_i(d\hat{x}_i\hat{x}_i)\right]
\]

后两项分别去除梯度的均值分量，以及梯度沿标准化输入方向的分量。

数值梯度检查结果：

```text
dx error     = 1.70e-09
dgamma error = 7.42e-13
dbeta error  = 2.88e-12
```

普通版与简化版 `dx` 的相对误差约为：

```text
2.03e-13
```

---

## 6. 接入 FullyConnectedNet

使用 BatchNorm 后，隐藏层结构为：

```text
Affine → BatchNorm → ReLU
```

反向传播顺序相反：

```text
ReLU → BatchNorm → Affine
```

每个隐藏层具有自己的 `gamma_i`、`beta_i` 和 `bn_param`。输出层保持：

```text
最后一个隐藏层 → Affine → Softmax
```

输出层不使用 BatchNorm，因为 Softmax 需要 logits 保留自由的相对尺度和偏移，而且预测不应依赖 batch 中的其他样本。

L2 正则化只作用于权重 `W`，不作用于 `b`、`gamma` 或 `beta`。

---

## 7. 深层网络收敛实验

在 1,000 个 CIFAR-10 样本、五个隐藏层的实验中：

| 指标 | BatchNorm | 无 BatchNorm |
|---|---:|---:|
| 最终 loss | 0.604 | 1.125 |
| 最佳训练准确率 | 84.3% | 55.7% |
| 最佳验证准确率 | 33.5% | 31.7% |

第 100 次迭代时：

```text
BatchNorm loss = 1.1711
Baseline loss  = 1.8035
```

BatchNorm 明显改善了优化条件并加快收敛，但训练准确率与验证准确率仍有很大差距，说明归一化不会自动消除过拟合。

---

## 8. 权重初始化尺度实验

无 BatchNorm 时：

- 权重过小，激活值和梯度经过多层连续相乘后逐渐趋近于零；
- 权重过大，激活、梯度或 logits 可能爆炸，造成训练和数值不稳定；
- 网络只能在相对有限的初始化范围内正常工作。

加入 BatchNorm 后，每层都会重新稳定激活尺度，因此网络在更宽的初始化范围内可以训练。

实验中的代表性验证准确率：

| `weight_scale` | 无 BatchNorm | BatchNorm |
|---:|---:|---:|
| `1e-4` | 11.9% | 26.2% |
| `1e-3` | 11.9% | 26.6% |
| `1e-2` | 21.4% | 29.1% |
| `1e-1` | 29.0% | 29.5% |
| `1` | 15.7% | 14.9% |

BatchNorm 只能降低初始化敏感性，不能补救任意极端的初始化。

---

## 9. Batch Size 实验

代表性结果：

| 模型 | Batch size | 最佳训练准确率 | 最佳验证准确率 |
|---|---:|---:|---:|
| 无 BatchNorm | 5 | 63.4% | 34.5% |
| BatchNorm | 5 | 42.5% | 31.1% |
| BatchNorm | 10 | 65.3% | 34.9% |
| BatchNorm | 50 | 87.8% | 34.9% |

batch 太小时，每个特征的均值和方差只由少量样本估计，统计噪声很大，BatchNorm 训练容易不稳定。batch 增大后，统计量更加可靠。

验证准确率不一定随 batch size 单调增加，因为小 batch 的优化噪声有时也具有正则化效果。

---

## 10. Layer Normalization

LayerNorm 不跨样本统计，而是对每个样本内部的全部特征统计：

```python
axis=1
keepdims=True
```

`keepdims=True` 使均值和方差保持 `(N, 1)`，从而可以沿特征维正确广播。

主要区别：

| | BatchNorm | LayerNorm |
|---|---|---|
| 归一化范围 | 同一特征跨样本 | 同一样本跨特征 |
| 统计轴 | `axis=0` | `axis=1` |
| 依赖 batch size | 是 | 否 |
| running statistics | 需要 | 不需要 |
| 训练/测试行为 | 不同 | 相同 |
| 常见场景 | CNN | RNN、Transformer |

LayerNorm 的简化反向公式与 BatchNorm 相同，但把样本数 `N` 换成特征数 `D`，求和轴改为 `axis=1`。

实现测试：

```text
LayerNorm dx error       = 2.11e-09
LayerNorm dgamma error   = 4.52e-12
LayerNorm dbeta error    = 2.58e-12
Full network max error   = 4.10e-07
```

LayerNorm 不依赖 batch 统计，但不同 batch size 仍会改变每个 epoch 的参数更新次数和优化噪声，因此训练曲线不一定完全相同。

---

## 11. 理论问题总结

在给出的数据预处理例子中：

- 从每张图片中减去数据集的平均图片，类似 BatchNorm：每个固定特征位置跨样本使用共享统计量；
- 把单张图片内全部 RGB 像素统一缩放到和为 1，类似 LayerNorm：每个样本使用自身全部特征的统计量。

LayerNorm 在特征维度很小时可能效果不好，因为用于估计均值和方差的数值太少，统计量不稳定、代表性较差。

---

## 12. 常见错误

- BatchNorm 错用 `axis=1`，或 LayerNorm 错用 `axis=0`。
- 方差使用 `N-1`，而不是本作业要求的总体方差。
- 把 `eps` 加在平方根外面。
- BatchNorm 测试阶段仍使用当前 batch 的统计量。
- 忘记更新或写回 running statistics。
- 输出层也加入归一化。
- 给 `gamma`、`beta` 添加 L2 正则化。
- LayerNorm 忘记 `keepdims=True`，造成广播方向错误。
