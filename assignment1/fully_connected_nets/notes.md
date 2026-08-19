# CS231n Assignment 1 — Fully Connected Networks 学习总结

## 1. 本次任务

本次作业将固定结构的两层神经网络推广为可以自由指定隐藏层数量的全连接网络，并实现三种常用优化算法：

```text
SGD + Momentum
RMSProp
Adam
```

完整实践流程为：

```text
参数初始化
→ 任意深度前向传播
→ Softmax Loss
→ 任意深度反向传播
→ 数值梯度检查
→ 小数据过拟合
→ 优化器实现与比较
→ CIFAR-10 最终训练
```

本作业暂时不实现 Batch/Layer Normalization 和 Dropout；它们属于 Assignment 2。

---

## 2. 任意深度网络的参数初始化

假设：

```python
hidden_dims = [H1, H2, ..., Hk]
```

完整维度列表可以写成：

```python
layer_dims = [input_dim] + hidden_dims + [num_classes]
```

第 `i` 个带参数层满足：

```python
Wi.shape = (layer_dims[i - 1], layer_dims[i])
bi.shape = (layer_dims[i],)
```

因此可以用一个循环生成全部参数：

```python
for i in range(1, self.num_layers + 1):
    self.params[f"W{i}"] = weight_scale * np.random.randn(
        layer_dims[i - 1], layer_dims[i]
    )
    self.params[f"b{i}"] = np.zeros(layer_dims[i])
```

权重使用高斯随机数打破神经元之间的对称性，偏置可以从零开始。

---

## 3. 任意深度前向传播

网络结构为：

```text
输入
→ Affine → ReLU
→ Affine → ReLU
→ ...
→ Affine
→ scores
```

前 `L-1` 层是隐藏层，统一调用：

```python
out, caches[i] = affine_relu_forward(out, Wi, bi)
```

最后一层只调用：

```python
scores, caches[L] = affine_forward(out, WL, bL)
```

最后一层不使用 ReLU，因为分类分数应当允许为正数或负数，随后直接送入 Softmax。

每层产生的 `cache` 都要保存，反向传播时会按照相反顺序使用。

---

## 4. Softmax、L2 正则化与反向传播

首先计算数据损失：

```python
loss, dout = softmax_loss(scores, y)
```

然后为所有权重加入 L2 正则化：

```python
loss += 0.5 * reg * np.sum(W * W)
```

其中 `0.5` 用来抵消平方项求导产生的 `2`，所以权重的正则化梯度为：

```python
reg * W
```

偏置不做正则化。

反向传播顺序与前向传播相反：

```text
Softmax
→ 输出层 Affine backward
→ 第 L-1 个隐藏层 Affine-ReLU backward
→ ...
→ 第 1 个隐藏层 Affine-ReLU backward
```

梯度使用与参数相同的键名保存：

```text
grads["W1"], grads["b1"], ..., grads["WL"], grads["bL"]
```

---

## 5. 数值梯度检查

梯度检查比较解析梯度和中心差分得到的数值梯度：

```text
dL/dW ≈ (L(W+h) - L(W-h)) / (2h)
```

本次 notebook 测试结果：

```text
reg = 0
Initial loss = 2.300479

W1 error = 1.03e-07
W2 error = 2.21e-05
W3 error = 4.56e-07
b1 error = 4.66e-09
b2 error = 2.09e-09
b3 error = 1.69e-10

reg = 3.14
Initial loss = 7.052115

W1 error = 3.90e-09
W2 error = 6.87e-08
W3 error = 3.48e-08
b1 error = 1.48e-08
b2 error = 1.46e-09
b3 error = 1.32e-10
```

`W2` 在无正则测试中的误差略大，通常是数值差分碰到 ReLU 在零点不可导造成的；其余结果以及加入正则后的结果都说明实现正确。

初始 loss 接近 `log(10) ≈ 2.3026`，符合 CIFAR-10 随机分类的预期。

---

## 6. 用 50 个样本做过拟合实验

小数据过拟合是一种重要的神经网络排错方法：

```text
如果模型连很少的训练样本都记不住，
通常说明实现、初始化或学习率存在问题。
```

### 三层网络

结构：

```python
hidden_dims = [100, 100]
```

有效参数：

```python
weight_scale = 1e-2
learning_rate = 1e-2
```

最终训练准确率约为：

```text
96%
```

### 五层网络

结构：

```python
hidden_dims = [100, 100, 100, 100]
```

有效参数：

```python
weight_scale = 6e-2
learning_rate = 2e-3
```

最终训练准确率：

```text
100%
```

五层网络对初始化尺度明显更加敏感。原来的 `weight_scale=1e-5` 经过多层连续相乘后，使激活值和梯度快速趋近于零，网络几乎无法学习。

---

## 7. SGD 与 Momentum

普通 SGD：

```python
w -= learning_rate * dw
```

Momentum 在当前梯度之外保留历史更新方向：

```python
v = momentum * v - learning_rate * dw
w = w + v
```

直观上可以把 `v` 看作参数移动的速度：

- 梯度长期方向一致时，速度逐渐积累，收敛更快；
- 梯度来回震荡时，不同方向会相互抵消，更新更加平稳。

实现测试：

```text
next_w error  = 8.88e-09
velocity error = 4.27e-09
```

---

## 8. RMSProp

RMSProp 保存梯度平方的指数移动平均：

```python
cache = decay_rate * cache + (1 - decay_rate) * dw**2
w -= learning_rate * dw / (np.sqrt(cache) + epsilon)
```

它为每个参数提供不同的有效学习率：

- 历史梯度较大的方向，分母较大，步长减小；
- 历史梯度较小的方向，分母较小，保留较大的更新步长。

`epsilon` 用来避免除零并提高数值稳定性。

实现测试：

```text
next_w error = 9.52e-08
cache error  = 2.65e-09
```

---

## 9. Adam

Adam 同时维护梯度的一阶矩和二阶矩：

```python
m = beta1 * m + (1 - beta1) * dw
v = beta2 * v + (1 - beta2) * dw**2
```

训练初期移动平均偏向零，因此需要偏差修正：

```python
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
```

参数更新：

```python
w -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
```

可以把 Adam 理解为：

```text
Momentum 的一阶动量
+ RMSProp 的二阶动量
+ 初期偏差修正
```

实现测试：

```text
next_w error = 1.14e-07
v error      = 4.21e-09
m error      = 4.21e-09
```

---

## 10. 优化器训练结果比较

使用相同的五层网络训练 5 个 epoch：

| 优化器 | 最终训练准确率 | 最终验证准确率 |
|---|---:|---:|
| SGD | 36.9% | 29.9% |
| SGD + Momentum | 51.9% | 35.2% |
| Adam | 57.4% | 35.6% |
| RMSProp | 53.3% | 37.4% |

可以观察到：

- 普通 SGD 在短时间内收敛最慢；
- Momentum 显著提升了 SGD 的收敛速度；
- RMSProp 和 Adam 可以按参数自适应调整步长；
- 单次短训练中 RMSProp 的验证准确率最高，不代表它在所有任务中都优于 Adam。

---

## 11. AdaGrad 为什么越训练越慢

AdaGrad 使用：

```python
cache += dw**2
w -= learning_rate * dw / (np.sqrt(cache) + epsilon)
```

因为 `cache` 永久累加所有历史梯度平方，它会单调增大，使有效学习率不断减小，最终可能导致更新非常缓慢。

Adam 使用梯度平方的指数移动平均，旧梯度的影响会逐渐衰减，因此通常不会出现同样严重的学习率持续衰减问题。

---

## 12. 最终 CIFAR-10 模型

最终使用三隐藏层网络：

```python
hidden_dims = [100, 100, 100]
weight_scale = 5e-2
reg = 2e-3
learning_rate = 8e-4
lr_decay = 0.95
num_epochs = 20
batch_size = 100
update_rule = "adam"
```

最终结果：

```text
Validation Accuracy = 52.7%
Test Accuracy       = 51.5%
```

两项都超过作业要求的 `50%`。

训练准确率最终约为 `70.9%`，明显高于验证准确率，说明模型已经出现一定程度的过拟合。后续的 Batch Normalization、Layer Normalization、Dropout 和卷积网络会进一步改善深层网络的训练与泛化。

---

## 13. 本次作业最重要的认识

### 深度增加并不保证更容易训练

网络越深，前向激活与反向梯度要经过更多次矩阵乘法，因此对初始化尺度和学习率更加敏感。

### 小数据过拟合是排错工具

在正式调参前，先确认模型能够记住少量样本，可以快速发现训练管线中的问题。

### 优化器只负责如何更新参数

模型的 `loss()` 负责前向传播、损失和梯度；`Solver` 调用优化器，根据梯度更新参数。模块职责分离后，可以在不修改模型的情况下切换 SGD、Momentum、RMSProp 或 Adam。

### 完整训练系统由多个部分共同决定

```text
网络结构
+ 权重初始化
+ Loss 与正则化
+ 正确的反向传播
+ 学习率
+ 优化器
+ 验证集调参
```

任何一部分不合适，都可能导致网络无法有效训练。

