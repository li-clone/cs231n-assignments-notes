# CS231n Assignment 1 — kNN 学习总结

## 1. kNN 基本思想

k-Nearest Neighbor 是一种非常直观的分类算法。

训练阶段几乎不进行计算，只是保存训练数据：

```python
self.X_train = X
self.y_train = y
```

预测时：

```text
测试样本
→ 与所有训练样本计算距离
→ 找最近的 k 个样本
→ 根据标签投票
→ 得到预测类别
```

其中 `k` 是人为选择的超参数。

---

## 2. CIFAR-10 与数据 Shape

CIFAR-10 图片大小为：

```text
32 × 32 × 3
```

展平后：

```text
3072 维向量
```

本次实验：

```text
X_train.shape = (5000, 3072)
X_test.shape  = (500, 3072)
```

最终距离矩阵：

```text
dists.shape = (500, 5000)
```

其中 `dists[i, j]` 表示第 `i` 个测试样本和第 `j` 个训练样本之间的距离。

---

## 3. L2 距离

本次主要使用欧氏距离：

\[
d(x,y)=\sqrt{\sum_i(x_i-y_i)^2}
\]

最直接的实现是：

```python
diff = X[i] - self.X_train[j]
squared_diff = diff ** 2
squared_distance = np.sum(squared_diff)
distance = np.sqrt(squared_distance)
```

---

## 4. Three Implementations

### Two Loops

```text
一个测试样本 × 一个训练样本
→ 一对一计算距离
```

逻辑最直观，但 Python 循环很多。

### One Loop

```text
一个测试样本 × 所有训练样本
→ 一次计算一整行距离
```

利用 NumPy broadcasting 减少了一层 Python 循环。

### No Loops

利用：

\[
\|x-y\|^2 = \|x\|^2 + \|y\|^2 - 2xy^T
\]

核心矩阵乘法：

```python
cross = X @ self.X_train.T
```

Shape：

```text
(500, 3072) @ (3072, 5000) = (500, 5000)
```

一次得到所有测试样本和训练样本之间的内积。

---

## 5. 向量化的理解

本次实际测试中大致出现：

```text
Two loops ≈ 14.5 s
One loop  ≈ 48 s
No loops  ≈ 0.4 s
```

说明：

> 减少 Python `for` 循环并不代表一定更快。

One-loop 会不断生成 `(5000, 3072)` 的大型临时数组，带来内存访问和分配开销。

No-loop 则把问题转化为高度优化的矩阵运算，因此速度提升非常明显。

真正重要的是：

```text
减少 Python 循环
+ 减少不必要的中间数据
+ 利用高效矩阵运算
```

---

## 6. axis

对于二维数组：

```text
axis=0 → 对每一列进行聚合
axis=1 → 对每一行进行聚合
```

例如：

```python
np.sum(X ** 2, axis=1)
```

表示对每个样本的 3072 个特征分别求平方和。

---

## 7. k 与超参数

`k` 是 Hyperparameter，而不是模型训练得到的 Parameter。

```text
Parameter：
W、b 等模型通过训练学习得到的参数

Hyperparameter：
k、learning rate、batch size、weight decay 等人为选择的参数
```

---

## 8. Cross Validation

为了选择合适的 `k`，使用 5-fold Cross Validation。

```text
5000 个训练样本
↓
分成 5 份
↓
每次 4 份训练 + 1 份验证
↓
轮流进行 5 次
```

对不同 `k` 分别计算验证集准确率，再选择表现最好的 `k`。

```text
Training Set   → 学习模型
Validation Set → 选择超参数
Test Set       → 最终评价泛化能力
```

不能使用 Test Set 来选择 `k`。

---

## 9. Final Result

本次最终选择：

```text
best_k = 12
```

在 500 张 CIFAR-10 测试图片上：

```text
141 / 500 correct
Accuracy = 0.282
```

即约 **28.2%**。

---

## 10. kNN 的特点

优点：

- 原理简单
- 几乎没有训练成本
- 适合理解距离、分类、验证集和超参数

缺点：

- 预测时需要和大量训练样本比较
- 训练集越大，推理越慢
- 原始像素距离不能很好表示图像语义
- 在高维数据上效果有限

kNN 使用人工定义的距离判断样本是否相似，而后续神经网络和 CNN 的重要能力之一，就是：

> 从数据中自动学习更加有意义的特征表示。

下一步将进入：

```text
Linear Classifier
→ Scores
→ Softmax Loss
→ Gradient
→ 参数更新
```
