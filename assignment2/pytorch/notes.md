# CS231n Assignment 2 — PyTorch 学习总结

## 1. 本次任务

本部分把前面手写的全连接层、卷积层、归一化、反向传播和参数更新迁移到 PyTorch，并依次使用三种抽象层级构建神经网络：

```text
Barebones Tensor API
→ nn.Module API
→ nn.Sequential API
→ CIFAR-10 Open-ended Challenge
```

最终目标是在 10 个 epoch 内，使 CIFAR-10 验证集准确率达到 70% 以上。

本次实际结果：

```text
Epoch 10 Validation Accuracy = 85.10%
```

---

## 2. PyTorch 的三个抽象层级

| 写法 | 参数管理 | 前向传播 | 参数更新 | 特点 |
|---|---|---|---|---|
| Barebones Tensor | 手动 | 手动编写函数 | 手动更新 | 最灵活，最接近底层原理 |
| `nn.Module` | 自动注册 | 自定义 `forward` | Optimizer | 灵活性与便利性兼顾 |
| `nn.Sequential` | 自动注册 | 按层顺序自动连接 | Optimizer | 最简洁，但不适合复杂分支 |

三个层级并不是三套不同原理。它们执行的仍然是：

```text
Forward → Loss → Backward → Update
```

区别只在于框架替我们管理了多少工作。

---

## 3. Tensor 与图片 Shape

PyTorch 中的图像 minibatch 通常使用 NCHW 排列：

```python
x.shape == (N, C, H, W)
```

- `N`：minibatch 大小；
- `C`：通道数；
- `H, W`：图像高度和宽度。

CIFAR-10 图片的形状为：

```text
(N, 3, 32, 32)
```

卷积层需要保留空间结构，而全连接层要求每个样本是一个向量。因此进入全连接层前需要 Flatten：

```python
def flatten(x):
    N = x.shape[0]
    return x.view(N, -1)
```

形状变化为：

```text
(N, C, H, W) → (N, C × H × W)
```

`-1` 表示让 PyTorch 根据张量元素总数自动推导该维度。

---

## 4. Barebones PyTorch

Barebones 写法直接使用 Tensor 和 `torch.nn.functional`，不定义模型类。

三层卷积网络结构：

```text
Conv 5×5, padding=2
→ ReLU
→ Conv 3×3, padding=1
→ ReLU
→ Flatten
→ Fully Connected
```

前向传播：

```python
x = F.relu(F.conv2d(x, conv_w1, bias=conv_b1, padding=2))
x = F.relu(F.conv2d(x, conv_w2, bias=conv_b2, padding=1))
x = flatten(x)
scores = x.mm(fc_w) + fc_b
```

最后不使用 Softmax：

```python
loss = F.cross_entropy(scores, y)
```

`F.cross_entropy` 已经把 LogSoftmax 与交叉熵组合起来，直接传入原始 scores 数值更稳定。

---

## 5. 卷积和全连接层的 Shape

PyTorch 卷积权重使用：

```text
(out_channels, in_channels, kernel_height, kernel_width)
```

本次 Barebones 参数为：

```python
conv_w1.shape == (32, 3, 5, 5)
conv_b1.shape == (32,)

conv_w2.shape == (16, 32, 3, 3)
conv_b2.shape == (16,)
```

两层卷积的 stride 都为 1，并使用保持尺寸的 padding：

```text
(N, 3, 32, 32)
→ (N, 32, 32, 32)
→ (N, 16, 32, 32)
```

Flatten 后：

```text
(N, 16 × 32 × 32)
```

矩阵乘法必须满足：

```text
(N, D) @ (D, 10) = (N, 10)
```

因此：

```python
fc_w.shape == (16 * 32 * 32, 10)
fc_b.shape == (10,)
```

---

## 6. Autograd 与计算图

创建参数时设置：

```python
w.requires_grad = True
```

涉及该参数的前向运算会被记录在动态计算图中。计算标量 loss 后调用：

```python
loss.backward()
```

PyTorch 会沿计算图反向传播，并把每个叶子参数的梯度保存在：

```python
w.grad
```

这意味着前几次作业中手写的局部梯度、链式法则和反向传播，现在由 Autograd 自动完成，但其数学原理没有改变。

---

## 7. 为什么必须清空梯度

PyTorch 默认累加参数梯度。连续执行两次 `backward()` 时：

```text
w.grad = gradient_batch_1 + gradient_batch_2
```

普通 minibatch SGD 每次只希望使用当前批次的梯度，因此更新后必须清空：

```python
with torch.no_grad():
    for w in params:
        w -= learning_rate * w.grad
        w.grad.zero_()
```

`zero_()` 末尾的下划线表示原地修改 Tensor。

使用 Optimizer 后通常写成：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

梯度累加并非错误行为。训练超大模型时，可以有意累加多个小批次的梯度来模拟更大的 batch；但普通训练必须显式清零。

---

## 8. `torch.no_grad()`

更新参数不属于网络需要求导的前向计算，因此手动更新权重时使用：

```python
with torch.no_grad():
    w -= learning_rate * w.grad
```

验证和测试时同样不需要梯度：

```python
with torch.no_grad():
    scores = model(x)
```

这样可以避免构建计算图，减少内存占用并提高推理速度。

---

## 9. Kaiming 初始化

ReLU 会把负数激活截断为 0。Kaiming 初始化根据输入连接数量调整权重方差，使多层网络的激活和梯度保持在较稳定的尺度：

\[
W \sim \mathcal{N}\left(0, \frac{2}{fan\_in}\right)
\]

Barebones 工具函数：

```python
w = torch.randn(shape, device=device, dtype=dtype) * np.sqrt(2.0 / fan_in)
w.requires_grad = True
```

`nn.Module` 中可以直接使用：

```python
nn.init.kaiming_normal_(layer.weight)
```

偏置通常从 0 开始：

```python
torch.zeros(shape, requires_grad=True)
```

---

## 10. 使用 `nn.Module`

自定义模型需要继承 `nn.Module`：

```python
class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, channel_1, 5, padding=2)
        self.conv2 = nn.Conv2d(channel_1, channel_2, 3, padding=1)
        self.fc = nn.Linear(channel_2 * 32 * 32, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = flatten(x)
        return self.fc(x)
```

层必须保存为 `self.conv1`、`self.fc` 等实例属性。这样 PyTorch 才会把子层与参数注册到模型中。

注册后，框架可以自动完成：

```python
model.parameters()   # 找到全部可学习参数
model.to(device)     # 移动整个模型
model.state_dict()   # 保存模型状态
model.train()        # 切换到训练模式
model.eval()         # 切换到评估模式
```

如果只在 `__init__` 中创建局部变量，函数结束后引用会丢失，PyTorch 也无法注册和优化其中的参数。

---

## 11. Optimizer

创建普通 SGD：

```python
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```

Optimizer 接收 `model.parameters()`，因此能够统一更新所有已注册层的权重和偏置。

标准训练循环：

```python
model.train()

scores = model(x)
loss = F.cross_entropy(scores, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

与 Barebones 的手动更新相比，Optimizer 只是封装了参数更新逻辑，反向传播仍由 Autograd 完成。

---

## 12. `train()` 与 `eval()`

训练前调用：

```python
model.train()
```

验证和测试前调用：

```python
model.eval()
```

这两个方法不会开启或关闭梯度，而是改变特定层的行为：

| 层 | Train 模式 | Eval 模式 |
|---|---|---|
| Dropout | 随机丢弃激活 | 不丢弃 |
| BatchNorm | 使用当前 batch 统计量 | 使用 running mean/var |

是否记录梯度由 `torch.no_grad()` 或 Tensor 的 `requires_grad` 控制。

---

## 13. 使用 `nn.Sequential`

当网络只是依次执行一组层时，可以使用：

```python
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=5, padding=2),
    nn.ReLU(),
    nn.Conv2d(32, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    Flatten(),
    nn.Linear(16 * 32 * 32, 10),
)
```

`nn.Sequential` 会把上一层输出自动传给下一层，因此不需要另外编写 `forward()`。

普通 Python 函数不能直接放入 `nn.Sequential`，因为其中的元素必须是 `nn.Module`。所以需要包装 Flatten：

```python
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)
```

Sequential 适合单一路径的前馈网络。具有残差连接、多分支、循环或多个输入输出的模型，通常需要自定义 `nn.Module.forward()`。

---

## 14. Momentum 与 Nesterov Momentum

普通 SGD 只使用当前梯度。Momentum 会保存历史更新形成的速度，使参数沿持续一致的方向加速，并减轻不同方向上的震荡。

本作业 Sequential 模型使用：

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=learning_rate,
    momentum=0.9,
    nesterov=True,
)
```

Nesterov Momentum 会先根据当前速度向前看一步，再在预计到达的位置计算修正方向，因此对参数变化趋势具有一定的预判性。

---

## 15. 开放挑战网络

最终模型结构：

```text
Input: 3 × 32 × 32

Conv 3×3, 64 → BatchNorm → ReLU
Conv 3×3, 64 → BatchNorm → ReLU
MaxPool 2×2

Conv 3×3, 128 → BatchNorm → ReLU
Conv 3×3, 128 → BatchNorm → ReLU
MaxPool 2×2

Conv 3×3, 256 → BatchNorm → ReLU
MaxPool 2×2

Flatten
Linear: 256 × 4 × 4 → 256
ReLU → Dropout(0.5)
Linear: 256 → 10
```

空间尺寸变化：

```text
32 × 32
→ MaxPool → 16 × 16
→ MaxPool → 8 × 8
→ MaxPool → 4 × 4
```

`2×2`、stride 2 的池化层把每个不重叠区域压缩成一个最大值，因此高度和宽度各减半。

Flatten 维度为：

```python
256 * 4 * 4
```

如果不使用池化，输入全连接层的空间位置数是 `32×32=1024`；三次池化后仅为 `4×4=16`，因此空间部分缩小了 64 倍。

---

## 16. 为什么最终网络有效

### 小卷积核

多个 `3×3` 卷积逐步提取复杂特征，同时比直接使用大卷积核更节省参数，并在层与层之间加入更多非线性。

### BatchNorm

稳定中间激活的尺度，使训练对初始化和学习率不那么敏感。

### Max Pooling

压缩空间尺寸，降低后续计算量，同时扩大深层神经元的有效感受野。

### Dropout

训练时随机关闭部分全连接激活，防止网络过度依赖少数神经元。

### Weight Decay

Adam 中设置：

```python
weight_decay=1e-4
```

它对较大的参数施加惩罚，起到正则化作用。

---

## 17. Adam 优化器

最终模型使用：

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)
```

Adam 同时维护梯度的一阶矩估计和二阶矩估计，为不同参数提供自适应更新尺度。在训练 epoch 较少的作业中，Adam 通常能较快得到可用结果。

这并不意味着 Adam 在所有任务上都优于 SGD。实际项目仍需要根据验证集结果选择优化器、学习率和调度策略。

---

## 18. 实验结果

本地使用 CPU、随机种子 0，对同一网络进行 10 个 epoch 的验证：

| Epoch | Average Loss | Validation Accuracy |
|---:|---:|---:|
| 1 | 1.5011 | 62.00% |
| 2 | 1.0980 | 70.90% |
| 3 | 0.9343 | 73.50% |
| 4 | 0.8194 | 77.10% |
| 5 | 0.7318 | 75.80% |
| 6 | 0.6629 | 80.70% |
| 7 | 0.5907 | 77.50% |
| 8 | 0.5327 | 82.20% |
| 9 | 0.4709 | 82.90% |
| 10 | 0.4144 | 85.10% |

第 2 个 epoch 已超过 70% 的要求，第 10 个 epoch 达到 85.10%。

验证准确率并非严格单调上升。例如第 5 和第 7 个 epoch 都出现了回落。这是 minibatch 随机性、参数更新和验证集泛化波动的正常现象。

实际训练应保存验证准确率最佳的 checkpoint，而不应默认最后一个 epoch 一定最好。

---

## 19. 常见易错点

### 1. 卷积权重维度写反

PyTorch 使用：

```text
(out_channels, in_channels, kH, kW)
```

### 2. 全连接输入维度计算错误

每次池化后都要重新计算 `C × H × W`，不能只看通道数。

### 3. 在 Cross Entropy 前手动 Softmax

`F.cross_entropy` 直接接收 scores，不需要提前 Softmax。

### 4. 忘记清空梯度

PyTorch 默认累加 `.grad`，正常训练必须调用 `optimizer.zero_grad()`。

### 5. 验证时忘记 `model.eval()`

这会让 Dropout 继续随机丢弃，并让 BatchNorm 使用当前 batch 的统计量。

### 6. 只调用 `model.eval()`，却忘记 `torch.no_grad()`

`eval()` 只改变层的行为，不会自动关闭梯度记录。

### 7. 在 `nn.Module` 中没有使用 `self.layer`

没有注册的层不会出现在 `model.parameters()` 中，Optimizer 也无法更新它们。

### 8. 把普通函数直接放入 Sequential

`nn.Sequential` 的成员必须是 `nn.Module`，普通函数需要先包装。

---

## 20. 本次知识链路

```text
手写 NumPy 网络
→ 理解每层 forward / backward
→ Tensor 记录动态计算图
→ Autograd 自动执行链式法则
→ nn.Module 注册和管理参数
→ Optimizer 统一更新参数
→ Sequential 简化线性拓扑
→ CNN + BatchNorm + Pooling + Dropout
→ CIFAR-10 Validation Accuracy 85.10%
```

PyTorch 并没有改变神经网络的数学原理，而是把已经理解的反向传播、参数管理、设备迁移和优化过程自动化。

---

## 21. 复习检查

完成本部分后，应能回答：

1. 为什么全连接层前需要 Flatten？
2. 为什么 `fc_w` 的形状要满足矩阵乘法规则？
3. `requires_grad=True` 的作用是什么？
4. 为什么 PyTorch 每次训练都要清空梯度？
5. `torch.no_grad()` 与 `model.eval()` 有什么区别？
6. 为什么自定义层要保存为 `self.layer`？
7. `nn.Module` 和 `nn.Sequential` 分别适合什么模型？
8. 三次 `2×2` 池化为什么把 `32×32` 变成 `4×4`？
9. BatchNorm 和 Dropout 在训练与测试阶段有什么区别？
10. 为什么最终实验应根据验证集选择模型，而不能调试时反复查看测试集？
