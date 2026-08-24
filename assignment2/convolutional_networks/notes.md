# CS231n Assignment 2 — Convolutional Networks 学习总结

## 1. 本次任务

本部分从零实现卷积和最大池化层，并将它们组合成三层卷积网络，最后实现适用于 CNN 特征图的 Spatial BatchNorm 与 GroupNorm。

完整实践流程为：

```text
Naive Convolution Forward
→ Convolution Backward
→ Max Pool Forward / Backward
→ Fast Layers
→ ThreeLayerConvNet
→ 数值梯度检查
→ 小数据过拟合
→ CIFAR-10 完整训练
→ Spatial BatchNorm
→ Spatial GroupNorm
```

---

## 2. 卷积输入与参数 Shape

输入图片使用 NCHW 排列：

```text
x.shape = (N, C, H, W)
```

- `N`：minibatch 中的图片数量；
- `C`：输入通道数；
- `H, W`：输入空间尺寸。

卷积核：

```text
w.shape = (F, C, HH, WW)
b.shape = (F,)
```

- `F`：卷积核数量，也是输出通道数；
- `HH, WW`：卷积核的空间尺寸；
- 每个卷积核覆盖全部 `C` 个输入通道。

输出：

```text
out.shape = (N, F, H_out, W_out)
```

---

## 3. 输出尺寸、Stride 与 Padding

设 padding 为 `P`，stride 为 `S`：

\[
H_{out}=1+\frac{H+2P-HH}{S}
\]

\[
W_{out}=1+\frac{W+2P-WW}{S}
\]

本作业保证结果可以整除。代码中使用整数除法：

```python
H_out = 1 + (H + 2 * pad - HH) // stride
W_out = 1 + (W + 2 * pad - WW) // stride
```

`stride` 决定相邻窗口在输入上的移动距离。`pad` 在输入边缘补零，可以控制输出尺寸并保留边缘信息。

当卷积核为奇数、`stride=1` 时，使用：

```python
pad = (filter_size - 1) // 2
```

可以保持输入输出的高度和宽度不变。

---

## 4. Naive Convolution Forward

只对空间维度 padding：

```python
x_pad = np.pad(
    x,
    ((0, 0), (0, 0), (pad, pad), (pad, pad)),
    mode="constant",
)
```

对每张图片、每个卷积核和每个输出位置遍历：

```python
h_start = i * stride
h_end = h_start + HH
w_start = j * stride
w_end = w_start + WW

window = x_pad[n, :, h_start:h_end, w_start:w_end]
out[n, f, i, j] = np.sum(window * w[f]) + b[f]
```

Shape 对齐：

```text
window.shape = (C, HH, WW)
w[f].shape   = (C, HH, WW)
```

深度学习框架中的“卷积”通常没有翻转卷积核，数学上更准确地说是 cross-correlation。只要训练和反向传播定义一致，这不会影响神经网络学习。

官方前向测试结果：

```text
Relative error = 2.21e-08
```

---

## 5. Convolution Backward

对单个输出位置定义：

```python
upstream = dout[n, f, i, j]
```

偏置被所有图片和空间位置共享：

```python
db = np.sum(dout, axis=(0, 2, 3))
```

卷积核梯度：

```python
dw[f] += window * upstream
```

输入窗口梯度：

```python
dx_pad[n, :, h_start:h_end, w_start:w_end] += w[f] * upstream
```

这里必须使用 `+=`：同一个输入像素可能同时被多个重叠窗口和多个卷积核使用，各条路径传来的梯度必须累加。

循环完成后裁掉 padding：

```python
if pad == 0:
    dx = dx_pad
else:
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
```

`pad=0` 必须单独处理，因为 `0:-0` 等价于 `0:0`，会得到空数组。

数值梯度检查：

```text
dx error = 6.25e-10
dw error = 8.90e-10
db error = 1.26e-11
```

---

## 6. Max Pooling

Max Pooling 对每张图片的每个通道独立处理，不混合通道，也没有可学习参数。

输出尺寸：

\[
H_{out}=1+\frac{H-PH}{S}
\]

\[
W_{out}=1+\frac{W-PW}{S}
\]

前向传播：

```python
window = x[n, c, h_start:h_end, w_start:w_end]
out[n, c, i, j] = np.max(window)
```

反向传播只把梯度传给前向最大值所在位置：

```python
mask = window == np.max(window)
dx[n, c, h_start:h_end, w_start:w_end] += (
    mask * dout[n, c, i, j]
)
```

重叠池化窗口仍然必须使用 `+=`。并列最大值处 Max 函数不可导；本作业把梯度传给所有并列最大位置。随机连续输入几乎不会出现并列，因此不影响数值梯度检查。

测试结果：

```text
Forward error           = 4.17e-08
Backward error          = 3.28e-12
Overlapping pool error  = 3.28e-12
```

---

## 7. Fast Layers 与 Sandwich Layers

四层 Python 循环易于理解，但训练速度很慢。课程提供的 `fast_layers.py` 使用 im2col 和 Cython 加速卷积与池化。

本地需要编译：

```text
python setup.py build_ext --inplace
```

常用组合层：

```text
conv_relu_forward
conv_relu_pool_forward
affine_relu_forward
```

组合层只是在内部顺序调用多个基础层，并把各层 cache 打包保存；反向传播再按相反顺序解包。

---

## 8. ThreeLayerConvNet 架构

网络结构：

```text
Conv → ReLU → 2×2 Max Pool
→ Affine → ReLU
→ Affine → Softmax
```

默认输入：

```text
(N, 3, 32, 32)
```

第一层卷积保持空间尺寸，`2×2`、`stride=2` 的池化将尺寸减半：

```text
(N, F, 32, 32)
→ (N, F, 16, 16)
```

参数 Shape：

```python
W1.shape = (num_filters, C, filter_size, filter_size)
b1.shape = (num_filters,)

flattened_dim = num_filters * (H // 2) * (W // 2)
W2.shape = (flattened_dim, hidden_dim)
b2.shape = (hidden_dim,)

W3.shape = (hidden_dim, num_classes)
b3.shape = (num_classes,)
```

默认配置中：

```text
W1 = (32, 3, 7, 7)
W2 = (8192, 100)
W3 = (100, 10)
```

---

## 9. 三层 CNN 前向与反向传播

前向传播使用三个模块：

```text
X
→ conv_relu_pool_forward   cache1
→ affine_relu_forward      cache2
→ affine_forward           cache3
→ scores
```

反向传播严格倒序：

```text
dscores
→ affine_backward
→ affine_relu_backward
→ conv_relu_pool_backward
```

数据损失使用 Softmax。三组权重都加入：

\[
\frac{1}{2}\lambda\sum W^2
\]

对应权重梯度增加：

```python
reg * W
```

偏置不进行 L2 正则化。

初始无正则损失：

```text
2.30258 ≈ log(10)
```

完整网络最大梯度误差：

```text
9.71e-06
```

低于 notebook 允许的 `1e-2`。

---

## 10. 小数据过拟合实验

使用 100 个训练样本进行排错：

```text
Initial loss          = 2.414
Final loss            = 0.556
Training accuracy     = 93.0%
Best validation acc   = 24.8%
```

10 epochs 已达到 93% 训练准确率，继续训练通常会更接近 100%。验证准确率较低是正常的，因为该实验的目的不是泛化，而是确认模型具有记忆小数据集的能力。

如果模型连很小的数据集都无法过拟合，通常应检查：

- 前向或反向传播是否错误；
- 学习率是否不合适；
- 权重初始化是否过大或过小；
- 网络容量是否不足；
- 正则化是否过强。

---

## 11. CIFAR-10 完整训练

使用完整 49,000 样本训练一轮：

```text
weight_scale  = 0.001
hidden_dim    = 500
reg           = 0.001
optimizer     = Adam
learning_rate = 0.001
batch_size    = 50
```

最终结果：

```text
Final loss          = 1.2751
Training accuracy   = 45.6%
Validation accuracy = 45.9%
```

超过 notebook 要求的约 40% 验证准确率。

---

## 12. Spatial Batch Normalization

普通 BatchNorm 输入为 `(N, D)`。CNN 特征图为：

```text
(N, C, H, W)
```

同一个通道由同一个卷积核产生，因此每个通道应该跨 `N、H、W` 统计均值和方差，不同通道独立。

可以通过变形直接复用普通 BatchNorm：

```text
(N, C, H, W)
→ transpose
(N, H, W, C)
→ reshape
(N*H*W, C)
```

调用普通 BatchNorm 后再恢复原形状。反向传播执行对称的 reshape 和 transpose。

梯度检查：

```text
dx error     = 5.61e-07
dgamma error = 7.88e-12
dbeta error  = 3.28e-12
```

---

## 13. Spatial Group Normalization

GroupNorm 把每个样本的 `C` 个通道分成 `G` 组：

```text
(N, C, H, W)
→ (N, G, C/G, H, W)
```

每组跨 `(C/G, H, W)` 计算统计量：

```python
axis=(2, 3, 4)
keepdims=True
```

每组的元素数量：

\[
M=(C/G)HW
\]

GroupNorm 不跨 batch 统计，所以：

- 不依赖 batch size；
- 不需要 running mean 或 running variance；
- 训练与测试行为相同。

两个特殊情况：

```text
G = 1 → 接近卷积版本 LayerNorm
G = C → 接近 InstanceNorm
```

梯度检查：

```text
dx error     = 4.99e-08
dgamma error = 8.18e-13
dbeta error  = 3.28e-12
```

---

## 14. 空间归一化对比

| 方法 | 每个归一化单元的统计范围 | 依赖 batch |
|---|---|---|
| Spatial BatchNorm | 同一通道跨 `N,H,W` | 是 |
| LayerNorm | 单个样本跨全部 `C,H,W` | 否 |
| GroupNorm | 单个样本的每组跨 `C/G,H,W` | 否 |
| InstanceNorm | 单个样本的每通道跨 `H,W` | 否 |

BatchNorm 在 batch 足够大时通常表现很好；GroupNorm 在检测、分割等受显存限制而只能使用小 batch 的任务中更有优势。

---

## 15. 常见错误

- 把 NCHW 与 NHWC 的轴顺序混淆。
- 输出尺寸公式忘记 `2 * pad` 或最后的 `+1`。
- padding 错加到 batch 或通道维度。
- 窗口起点忘记乘 `stride`。
- 卷积反向对 `dx`、`dw` 使用赋值而不是累加。
- `pad=0` 时仍使用 `pad:-pad` 裁剪。
- Max Pool 反向把梯度传给整个窗口，而不是最大值位置。
- 三层 CNN 的 `W2` 输入维度忘记考虑池化后的空间尺寸。
- 输出层错误地加入 ReLU。
- 给偏置加入 L2 正则化。
- Spatial BatchNorm 错误地分别归一化每个空间位置。
- GroupNorm 的 `C` 不能被 `G` 整除。
- GroupNorm 对 batch 维求统计，失去其不依赖 batch 的特性。

---

## 16. 核心记忆

```text
卷积前向：窗口 × 卷积核，求和后加偏置
卷积反向：dout 分别累加到 db、dw、dx
最大池化：只保留窗口最大值
池化反向：梯度只传给最大值位置
CNN：局部连接 + 参数共享
Spatial BatchNorm：每通道跨 N,H,W
GroupNorm：每样本每组跨 C/G,H,W
```
