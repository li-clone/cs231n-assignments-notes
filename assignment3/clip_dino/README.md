# CLIP and DINO

状态：代码、理论问答和真实模型实验均已完成。

对应课程 Notebook：`CLIP_DINO.ipynb`。

已完成内容：

- 向量化图文余弦相似度矩阵
- CLIP Zero-Shot Classification
- 缓存图片特征的文本到图片检索
- DINO ViT token shape、attention 与 patch PCA 可视化
- 冻结 DINO、使用单帧标注训练线性 patch 分类器
- DAVIS `soapbox` 视频 99 帧完整分割与 IoU 评估

真实实验：

```text
CLIP model                  = ViT-B/32
CLIP zero-shot agreement    = 9 / 10
DINO model                  = ViT-S/8
DAVIS video                 = soapbox, 99 frames, 4 classes
Segmentation training frame = frame 40
Training-frame IoU          = 1.0000
All-frame mean IoU          = 0.5609
Required mean IoU           > 0.55
```

目录内容：

- `CLIP_DINO.ipynb`：课程完整实验流程
- `clip_dino.py`：CLIP 分类、检索及 DINO 线性分割实现
- `clip_zero_shot_predictions.png`：CLIP 零样本分类结果
- `clip_text_retrieval.png`：CLIP 文本检索结果
- `dino_attention_heads.png`：DINO 多头注意力可视化
- `dino_patch_pca.png`：DINO patch features 的 PCA 可视化
- `dino_segmentation_comparison.png`：三帧分割对比
- `dino_segmentation_comparison.mp4`：99 帧完整分割视频
- `dino_segmentation_metrics.json`：真实分割指标
- `notes.md`：中文学习总结

详细学习总结见 [notes.md](notes.md)。
