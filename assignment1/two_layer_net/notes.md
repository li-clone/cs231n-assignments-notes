# CS231n Assignment 1 — Two-Layer Neural Network 学习总结

## 1. 两层神经网络整体结构

本次作业从线性分类器进一步扩展到了一个带隐藏层的神经网络。

完整结构：

```text
输入 X
→ Affine 1
→ ReLU
→ Affine 2
→ Softmax Loss
```

也可以写成：

```text
X → hidden → scores → loss
```

这里称为 Two-Layer Network，是因为通常只计算带可学习参数的层：

```text
第一层：Affine，参数为 W1、b1
第二层：Affine，参数为 W2、b2
```

ReLU 没有需要学习的参数，因此通常不计入层数。

---

## 2. 数据与参数 Shape

CIFAR-10 图片大小：

```text
3 × 32 × 32
```

每张图片展平后的维度：

```text
D = 3 × 32 × 32 = 3072
```

如果隐藏层大小为 `H`，类别数为 `C=10`，则参数 Shape 为：

```text
X  : (N, 3072)
W1 : (3072, H)
b1 : (H,)
W2 : (H, 10)
b2 : (10,)
```

数据经过网络后的 Shape 变化：

```text
X       : (N, 3072)
hidden  : (N, H)
scores  : (N, 10)
```

---

## 3. 模块化 Layer 设计

本次作业没有把整个网络写成一个大函数，而是先实现每一层的：

```text
forward
backward
```

前向传播函数返回：

```python
out, cache = layer_forward(x, ...)
```

其中：

- `out`：当前层的输出，交给下一层；
- `cache`：保存反向传播需要的输入和参数。

反向传播函数接收：

```python
dx, dw, db = layer_backward(dout, cache)
```

`dout` 是从网络后方传来的上游梯度，`cache` 则提供当前层求导所需的信息。

这种模块化设计使不同网络可以像搭积木一样组合：

```text
Affine
Affine + ReLU
Affine + ReLU + Affine
更深的 Fully Connected Network
```

---

## 4. Affine Layer

### Forward

全连接层前向传播公式：

\[
out=XW+b
\]

输入 `x` 可能是多维图片数据，因此先将每个样本展平：

```python
x_row = x.reshape(x.shape[0], -1)
out = x_row.dot(w) + b
```

偏置 `b` 的 Shape 为 `(M,)`，NumPy 会利用 broadcasting 将它加到每个样本上。

### Backward

上游梯度为 `dout` 时：

\[
dW=X^Tdout
\]

\[
db=\sum_{i=1}^{N}dout_i
\]

\[
dX=doutW^T
\]

对应实现：

```python
dw = x_row.T.dot(dout)
db = np.sum(dout, axis=0)
dx_row = dout.dot(w.T)
dx = dx_row.reshape(x.shape)
```

最后把 `dx` reshape 回原始输入 Shape，是因为前向传播时曾经将输入展平。

---

## 5. ReLU Layer

ReLU 定义：

\[
ReLU(x)=\max(0,x)
\]

前向传播：

```python
out = np.maximum(0, x)
```

反向传播：

```python
dx = dout * (x > 0)
```

含义：

```text
x > 0 → 梯度正常通过
x < 0 → 梯度为 0
```

ReLU 为网络引入非线性。如果没有 ReLU，多层 Affine 仍然可以合并成一次线性变换，增加层数也无法明显增强表达能力。

---

## 6. 激活函数与梯度消失

### Sigmoid

Sigmoid 导数：

\[
\sigma'(x)=\sigma(x)(1-\sigma(x))
\]

当输入是绝对值很大的正数或负数时，Sigmoid 会进入饱和区，导数接近 0，因此容易出现梯度消失。

### ReLU

当 `x<0` 时，ReLU 的导数为 0。如果一个神经元长期落在负数区域，它可能无法继续更新，这就是死亡 ReLU。

### Leaky ReLU

Leaky ReLU 在负数区域保留一个小斜率：

\[
f(x)=\begin{cases}
x, & x>0 \\
\alpha x, & x\leq0
\end{cases}
\]

其负数区域的导数为正常数 `α`，因此可以缓解死亡 ReLU 问题。

---

## 7. Sandwich Layer

神经网络中经常连续使用：

```text
Affine → ReLU
```

因此作业提供了组合函数：

```python
hidden, cache = affine_relu_forward(X, W1, b1)
```

反向传播：

```python
dx, dW1, db1 = affine_relu_backward(dhidden, cache)
```

它没有引入新的数学逻辑，只是把常用层组合起来，减少重复代码，并让网络结构更加清晰。

组合层的梯度检查结果：

```text
dx error: 2.30e-11
dw error: 8.16e-11
db error: 7.83e-12
```

---

## 8. Softmax Loss Layer

第二层 Affine 输出分类分数：

```text
scores.shape = (N, C)
```

Softmax 将分数转换为概率：

\[
p_{i,j}=\frac{e^{s_{i,j}}}{\sum_k e^{s_{i,k}}}
\]

为避免数值溢出，先减去每个样本的最大分数：

```python
scores = x - np.max(x, axis=1, keepdims=True)
```

Cross Entropy Loss：

\[
L=-\frac{1}{N}\sum_i\log p_{i,y_i}
\]

Softmax 与 Cross Entropy 合并后的梯度：

```python
dx = probs.copy()
dx[np.arange(N), y] -= 1
dx /= N
```

本次测试：

```text
Softmax loss: 2.3025458
Gradient error: 8.23e-09
```

CIFAR-10 有 10 类，随机初始化时初始 loss 接近：

\[
-\log(0.1)=2.3026
\]

---

## 9. TwoLayerNet 参数初始化

网络参数保存在：

```python
self.params
```

初始化方式：

```python
self.params["W1"] = weight_scale * np.random.randn(input_dim, hidden_dim)
self.params["b1"] = np.zeros(hidden_dim)
self.params["W2"] = weight_scale * np.random.randn(hidden_dim, num_classes)
self.params["b2"] = np.zeros(num_classes)
```

权重不能全部初始化为 0。否则同一层的神经元会得到相同输出和相同梯度，无法打破对称性。

偏置可以初始化为 0，因为随机权重已经打破了神经元之间的对称性。

---

## 10. TwoLayerNet Forward Pass

完整前向传播：

```python
hidden, cache1 = affine_relu_forward(X, W1, b1)
scores, cache2 = affine_forward(hidden, W2, b2)
```

其中：

```text
cache1 → 保存第一层 Affine 和 ReLU 的反向传播信息
cache2 → 保存第二层 Affine 的反向传播信息
```

如果 `y is None`，说明当前处于预测模式，只返回：

```python
return scores
```

如果传入标签 `y`，则继续计算 loss 和梯度，进入训练模式。

---

## 11. TwoLayerNet Backward Pass

反向传播必须按照前向传播的相反顺序执行：

```text
前向：Affine-ReLU → Affine → Softmax
反向：Softmax → Affine backward → Affine-ReLU backward
```

代码主线：

```python
loss, dscores = softmax_loss(scores, y)

dhidden, dW2, db2 = affine_backward(dscores, cache2)
dX, dW1, db1 = affine_relu_backward(dhidden, cache1)
```

最后将梯度保存到与参数同名的字典中：

```python
grads["W1"] = dW1
grads["b1"] = db1
grads["W2"] = dW2
grads["b2"] = db2
```

这体现了链式法则：梯度从 loss 出发，沿计算图逐层向前传递。

---

## 12. L2 Regularization

本次网络的总损失为：

\[
L=L_{data}+\frac{1}{2}reg(\|W_1\|^2+\|W_2\|^2)
\]

代码：

```python
loss += 0.5 * self.reg * (
    np.sum(W1 * W1) + np.sum(W2 * W2)
)
```

因为损失中包含 `0.5`，求导后正好得到：

\[
\frac{\partial L_{reg}}{\partial W}=regW
\]

所以：

```python
grads["W1"] = dW1 + self.reg * W1
grads["W2"] = dW2 + self.reg * W2
```

偏置 `b1`、`b2` 通常不做正则化。

Regularization 的作用不是直接提高训练准确率，而是限制模型过度依赖某些权重，从而改善泛化能力。

---

## 13. Numerical Gradient Check

反向传播容易出现符号、转置和 Shape 错误，因此训练前使用 Numerical Gradient Check：

\[
\frac{\partial L}{\partial W}\approx
\frac{L(W+h)-L(W-h)}{2h}
\]

然后比较：

```text
解析梯度 vs 数值梯度
```

本次两层网络的测试结果：

```text
无正则 loss error: 4.61e-12
有正则 loss error: 3.86e-11

W1 gradient error: 约 1e-8 ～ 1e-7
W2 gradient error: 约 1e-10 ～ 1e-8
b1 gradient error: 约 1e-9 ～ 1e-8
b2 gradient error: 约 1e-10
```

这些误差都非常小，说明前向损失和反向梯度实现正确。

---

## 14. Solver 与训练流程

`TwoLayerNet` 只负责：

```text
Forward
Loss
Backward
```

`Solver` 负责真正的训练循环：

```text
随机抽取 minibatch
→ 调用 model.loss(X_batch, y_batch)
→ 得到 loss 和 grads
→ 使用 SGD 更新参数
→ 每个 epoch 检查准确率
→ 保存 validation accuracy 最高的参数
```

本次训练参数：

```text
hidden_size  = 50
learning_rate = 1e-3
lr_decay      = 0.95
num_epochs    = 10
batch_size    = 100
update_rule   = SGD
```

训练集有 49,000 张图片：

```text
49000 / 100 = 490 iterations per epoch
490 × 10 = 4900 total iterations
```

`lr_decay=0.95` 表示每完成一个 epoch：

```python
learning_rate *= 0.95
```

学习率逐渐减小，可以让训练前期快速前进、后期更稳定地收敛。

---

## 15. Loss 与 Accuracy 曲线

初始 loss 约为：

```text
2.303
```

训练后逐渐下降到大约：

```text
1.1 ～ 1.4
```

Loss 曲线整体下降，说明网络确实在学习。曲线中存在上下波动是正常现象，因为每个 iteration 使用的 minibatch 不同。

Accuracy 曲线中：

- 训练准确率与验证准确率整体共同上升；
- 两条曲线差距不大，原始模型没有严重过拟合；
- 验证准确率最高出现在约第 8 个 epoch。

`Solver` 训练结束后会自动恢复 validation accuracy 最高时的参数，而不是简单保留最后一个 epoch 的参数。

---

## 16. 第一层权重可视化

将 `W1` 的每个隐藏神经元重新 reshape 为：

```text
3 × 32 × 32
```

可以观察到一些颜色块、明暗区域和低频图像结构。

这些权重还不像 CNN 卷积核那样清晰，原因包括：

- 全连接层直接连接整张图片；
- 不利用图像的局部空间结构；
- 对物体位置变化不够鲁棒；
- 每个隐藏单元需要同时处理整幅图像。

这说明两层全连接网络已经能自动学习特征，但其视觉归纳偏置仍然弱于 CNN。

---

## 17. 控制变量调参

为了观察模型容量的影响，只改变隐藏层大小：

```text
实验一：hidden_size = 50
实验二：hidden_size = 100
```

其他参数保持不变。

结果：

```text
hidden_size = 50
best validation accuracy = 0.527

hidden_size = 100
best validation accuracy = 0.518
最高训练准确率约为 0.620
```

隐藏层增大后，训练准确率提高，但验证准确率下降。

这说明：

```text
模型容量增大
→ 更容易拟合训练数据
→ 训练准确率提高
→ 泛化能力没有同步提高
→ 过拟合差距变大
```

因此不能只根据训练准确率选择模型，应该使用 validation accuracy 选择超参数。

最终保留 `hidden_size=50` 的模型作为 `best_model`。

---

## 18. 如何减小训练与测试准确率差距

如果训练准确率明显高于测试准确率，说明模型可能过拟合。

有效方法包括：

### 使用更大的训练数据集

更多数据能够覆盖更多样本变化，使模型不容易只记住现有训练集。

### 增大正则化强度

更强的正则化能够限制权重过度拟合训练数据，但过强也可能造成欠拟合。

### 不应盲目增加隐藏单元

增加隐藏单元会提高模型容量，通常会进一步降低训练误差，但也可能加重过拟合。

因此 Inline Question 2 的选择是：

```text
1. Train on a larger dataset.
3. Increase the regularization strength.
```

---

## 19. Final Result

本次最终模型：

```text
hidden_size = 50
learning_rate = 1e-3
lr_decay = 0.95
num_epochs = 10
batch_size = 100
```

最终结果：

```text
Validation Accuracy = 0.527
Test Accuracy       = 0.496
```

即：

```text
验证集准确率：52.7%
测试集准确率：49.6%
```

两项都超过作业要求的 48%。

最佳模型已经保存为：

```text
best_two_layer_net.npy
```

---

## 20. 本次作业真正掌握的主线

这次作业第一次完整走通了一个神经网络分类器的全部流程：

```text
实现基础 Layer
→ 保存 forward cache
→ 使用链式法则进行 backward
→ 用 Numerical Gradient Check 验证梯度
→ 将 Layer 组合成 TwoLayerNet
→ 使用 Softmax + L2 计算 Loss
→ 使用 Solver 和 SGD 训练
→ 绘制 Loss / Accuracy 曲线
→ 判断过拟合与欠拟合
→ 使用 Validation Set 调参
→ 在 Test Set 上做最终评价
```

相比 Softmax 线性分类器，两层神经网络多了：

```text
隐藏层
+ ReLU 非线性
+ 分层特征表示
+ 更强的模型容量
```

最重要的认识是：

> 神经网络训练并不只是写出 Forward。一个可靠的训练系统还需要正确的 Backward、梯度检查、优化器、验证集调参和泛化能力分析。

这套模块化思路会直接延伸到后续的多层全连接网络和卷积神经网络：

```text
Layer Forward
→ Cache
→ Loss
→ Layer Backward
→ Gradient
→ Parameter Update
```
