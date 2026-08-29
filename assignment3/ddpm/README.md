# Denoising Diffusion Probabilistic Models

状态：代码、理论问答和预训练模型采样均已完成。

对应课程 Notebook：`DDPM.ipynb`。

已完成内容：

- 前向扩散采样 `q_sample`
- 在 `x_t`、`x_0` 和噪声之间进行代数转换
- 带时间与文本条件的 U-Net
- 下采样、上采样与 Skip Connection
- DDPM 去噪训练损失 `p_losses`
- 单步反向采样 `p_sample`
- Classifier-Free Guidance
- 加载第 70,000 步预训练权重并生成 Emoji

实际生成实验：

```text
Prompt               = face with cowboy hat
Checkpoint step      = 70,000
Reverse steps        = 100
Batch size           = 5
Device               = NVIDIA RTX 4070 Laptop GPU
Sampling time        ≈ 5 seconds
```

验证情况：

- `q_sample` 与代数转换误差均小于 `2e-6`
- U-Net 参数数量测试通过：`6499`
- U-Net 输出尺寸、有限值和反向传播测试通过
- `p_losses` 的两种 objective 与手工加权 MSE 完全一致
- `p_sample` 的两种 objective、`t>0` 和 `t=0` 均与手算完全一致
- CFG 确定性公式测试通过，且不会修改调用者传入的参数字典

课程 Notebook 的部分固定随机数组测试与本地 PyTorch 2.8 存在差异，原因是训练态 Dropout 的随机序列随 PyTorch 版本发生变化。确定性结构、公式与梯度测试均已单独通过。

目录内容：

- `DDPM.ipynb`：课程完整实验流程
- `gaussian_diffusion.py`：前向扩散、训练损失与反向采样
- `unet.py`：条件 U-Net 与 Classifier-Free Guidance
- `ddpm_face_with_cowboy_hat_final.png`：最终生成的 5 个 Emoji
- `ddpm_face_with_cowboy_hat_process.png`：从噪声到 Emoji 的采样过程
- `README.md`：完成状态与结果摘要
- `notes.md`：中文学习总结

详细学习总结见 [notes.md](notes.md)。
