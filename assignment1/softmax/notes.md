# CS231n Assignment 1 — Softmax 学习总结

## 1. Softmax 分类器整体流程

Softmax 是一个线性分类器，核心流程：

```text
输入 X
→ scores = XW
→ Softmax 概率
→ Cross Entropy Loss
→ 反向传播求梯度 dW
→ SGD 更新 W
```

其中：

- `X`：输入数据
- `W`：模型真正学习的权重参数
- `scores`：每个类别的打分
- `loss`：模型当前错得有多严重
- `dW`：loss 对 W 的梯度，告诉 W 应该怎么修改

---

## 2. 权重 W 与类别分数

线性分类器使用：

\[
scores = XW
\]

若：

```text
X.shape = (N, D)
W.shape = (D, C)
```

则：

```text
scores.shape = (N, C)
```

`scores[i, j]` 表示第 `i` 个样本对第 `j` 个类别的分数。

训练的本质就是不断调整 `W`，使正确类别的 score 更高。

---

## 3. Softmax 与 Cross Entropy

Softmax 将 scores 转换成概率：

\[
p_j=\frac{e^{s_j}}{\sum_k e^{s_k}}
\]

为了防止 `exp()` 数值溢出，通常先做：

```python
scores -= np.max(scores)
```

Cross Entropy Loss：

\[
L=-\log(p_{\text{correct}})
\]

正确类别概率越高，loss 越小。

CIFAR-10 有 10 类，模型刚随机初始化时每类概率约为：

\[
0.1
\]

因此初始 loss 应接近：

\[
-\log(0.1)\approx2.3026
\]

---

## 4. Loss、Gradient 与 W 的关系

正向传播：

```text
W
→ scores
→ probability
→ loss
```

反向传播：

```text
loss
→ dscores
→ dW
```

Softmax + Cross Entropy 有一个重要结果：

\[
\frac{\partial L}{\partial scores}=p-y
\]

代码中：

```python
dscores = p.copy()
dscores[y[i]] -= 1
```

含义：

```text
正确类别 → 梯度为 p - 1，推动 score 增大
错误类别 → 梯度为 p，推动 score 减小
```

然后通过：

\[
scores=XW
\]

继续反向得到：

\[
dW=X^Tdscores
\]

`dW` 与 `W` shape 相同，因为 W 中每一个参数都需要一个对应梯度。

---

## 5. Naive 与 Vectorized

### Naive

一张图片一张图片计算：

```text
X[i]
→ scores
→ probability
→ loss
→ dscores
→ dW
```

每个样本的梯度使用：

```python
dW += np.outer(X[i], dscores)
```

### Vectorized

一次处理整个 batch：

```python
scores = X.dot(W)
```

最终梯度：

```python
dW = X.T.dot(dscores)
```

数学逻辑完全相同，只是利用矩阵运算替代 Python 循环，速度更高。

---

## 6. Regularization

总 loss 不只考虑分类错误：

\[
L=L_{data}+reg\sum W^2
\]

对应梯度：

\[
dW=dW_{data}+2regW
\]

`reg` 的作用是限制 W 变得过大，降低模型过拟合的风险。

---

## 7. SGD 与 Mini-batch

训练时不需要每一步都使用整个训练集，而是随机抽取一小批数据：

```python
batch_indices = np.random.choice(num_train, batch_size)
X_batch = X[batch_indices]
y_batch = y[batch_indices]
```

每一步：

```text
随机抽 mini-batch
→ 计算 loss
→ 计算 grad
→ 更新 W
→ 重复
```

权重更新公式：

\[
W \leftarrow W-\eta dW
\]

代码：

```python
self.W -= learning_rate * grad
```

其中 `learning_rate` 控制每次更新的步长。

---

## 8. 两个重要超参数

### Learning Rate

控制：

> 每次沿梯度方向修改 W 的幅度。

太小：训练慢。  
太大：可能震荡甚至发散。

### Regularization Strength

控制：

> 对大权重的惩罚强度。

`learning_rate` 决定“每一步走多远”，  
`reg` 决定“多强地把 W 往较小的方向约束”。

两者都需要通过 validation set 调整，而不是由模型自动学习。

---

## 9. Validation 与超参数搜索

尝试不同组合：

```text
learning_rate × regularization_strength
```

每组参数都：

```text
训练 Softmax
→ 计算 training accuracy
→ 计算 validation accuracy
→ 保存结果
```

最终选择 validation accuracy 最高的模型。

这里再次强化：

```text
Training Set   → 学习 W
Validation Set → 选择超参数
Test Set       → 最终评价泛化能力
```

---

## 10. Predict

训练结束后预测：

```python
scores = X.dot(self.W)
y_pred = np.argmax(scores, axis=1)
```

即：

> 对每张图片选择 score 最大的类别。

---

## 11. 可视化 W 的理解

将每个类别对应的权重重新 reshape 成图像后，会看到模糊的类别模板。

例如：

- frog 可能偏绿色
- ship 常出现蓝色背景
- truck 会出现模糊车辆轮廓

原因是线性分类器每个类别只有一套权重模板，需要同时适配不同姿态、背景和位置，因此只能学习比较粗糙的全局模式。

这也说明线性分类器表达能力有限，而 CNN 能进一步学习局部、层次化的视觉特征。

---

## 12. Softmax 与 SVM 的一个区别

SVM 使用 hinge loss：

\[
\max(0, s_j-s_y+\Delta)
\]

如果样本已经分类得“足够好”，SVM loss 可以直接变成 0。

Softmax Cross Entropy 即使已经预测正确，只要正确类别概率还没有达到 1，loss 通常仍然大于 0。

因此：

```text
SVM：
分得足够好 → 不再继续惩罚

Softmax：
分对之后 → 仍继续鼓励模型提高正确类别的置信度
```

---

## 13. 本次作业真正掌握的主线

这部分第一次完整实现了一个真正可以训练的分类器：

```text
数据 X
→ 权重 W
→ scores
→ Softmax
→ Cross Entropy Loss
→ gradient
→ SGD
→ 更新 W
→ validation 调参
→ prediction
```

相比 kNN 只是“记住训练数据”，Softmax 开始真正通过优化算法学习模型参数 `W`。

这也是后面神经网络训练的基础：

```text
Forward
→ Loss
→ Backpropagation
→ Gradient
→ Parameter Update
```
