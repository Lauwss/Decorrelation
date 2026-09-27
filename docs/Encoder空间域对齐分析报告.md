# Encoder 空间域对齐分析报告：NCE 主分析与 raw Encoder 辅助证据

> 本报告的主分析空间是训练实际使用的 PatchNCE/Expander feature space；raw Encoder 结果只作为 additional evidence。

## 0. 结论口径

由于 PatchNCE 的正样本约束和 VCReg 去相关约束都围绕 Expander 后的 feature 展开，模型域对齐能力的主要判断应以 NCE/Expander 空间为准。raw Encoder 的测量用于补充回答：观察到的域关系是否在进入 Expander 之前已经存在，以及 VCReg 的影响是否可能沿 Encoder 深度向前传递。

## 1. 实验问题

本实验不加入 ASM、模糊、gamma、噪声或对比度等退化，只复现此前 NCE 诊断中的三组关系，并将比较位置前移到 Generator 的 raw Encoder：

- $A$：`testA` 中的输入图像；
- $B$：同名配对的 `testB` 清晰图像；
- $G(A)$：当前 checkpoint 对 $A$ 的生成结果。

目标是先在 Expander/NCE 空间判断 VCReg 是否改变 $A/B/G(A)$ 的域关系，再用 raw Encoder 结果判断这种变化是否也能在投影前观察到。

## 2. 实验协议

- 数据：`datasets/RRSHID/testA` 与 `testB` 的 304 对同名 PNG；
- 模型：`cut_baseline` 与 `expander_decorr_200_exp1_s1407`；
- checkpoint：两个实验目录下的 `best_test_psnr.pt`；
- checkpoint epoch：CUT 为 165，VCReg 为 195；
- 预处理：RGB，像素归一化为 `pixel / 127.5 - 1.0`，反射填充到 4 的倍数；
- Encoder 层：`stem`、`downsample_1`、`downsample_2`、`enc_resblock_1`、`enc_resblock_6`；
- 位置协议：每一层只比较相同空间位置的 raw feature 向量；
- 结果行数：$304 \times 2 \times 5=3040$；
- 未使用任何退化模块，也未重新训练模型。

对每个模型，生成路径为：

```text
bottleneck, features_A = generator.encode(A, return_features=True)
G(A) = generator.decode(bottleneck)
features_G = generator.encode(G(A), return_features=True)
```

## 3. Raw Encoder 辅助证据的距离定义

对同一层、同一空间位置的 raw Encoder 向量 $f_1$ 和 $f_2$，主指标为 cosine distance：

$$
d_{\cos}(f_1,f_2)=1-
\frac{f_1^\top f_2}{\lVert f_1\rVert_2\lVert f_2\rVert_2}
$$

三组主距离为：

$$
d(A,B),\quad d(A,G(A)),\quad d(B,G(A))
$$

源域偏好定义为：

$$
\Delta_{source}=d(B,G(A))-d(A,G(A))
$$

$\Delta_{source}>0$ 表示 $G(A)$ 更接近输入源域 $A$；数值越大，表示生成结果在该 Encoder 层越明显地保留源域关系。另记录 normalized L2 和 feature norm ratio，用于排除纯粹的 feature scale 变化。

## 4. Raw Encoder 全量结果

下表为 304 对图像的 cosine distance 均值。

| 模型 | 层 | $d(A,B)$ | $d(A,G(A))$ | $d(B,G(A))$ | $\Delta_{source}$ |
| --- | --- | ---: | ---: | ---: | ---: |
| CUT | `stem` | 0.498303 | 0.147834 | 0.453242 | 0.305408 |
| CUT | `downsample_1` | 0.324440 | 0.075384 | 0.294911 | 0.219528 |
| CUT | `downsample_2` | 0.232344 | 0.058577 | 0.206586 | 0.148009 |
| CUT | `enc_resblock_1` | 0.501677 | 0.137340 | 0.460681 | 0.323341 |
| CUT | `enc_resblock_6` | 0.682197 | 0.177999 | 0.625193 | 0.447194 |
| VCReg | `stem` | 0.474143 | 0.149466 | 0.427154 | 0.277688 |
| VCReg | `downsample_1` | 0.336316 | 0.091950 | 0.302214 | 0.210264 |
| VCReg | `downsample_2` | 0.259877 | 0.076859 | 0.229798 | 0.152939 |
| VCReg | `enc_resblock_1` | 0.485281 | 0.164511 | 0.433743 | 0.269232 |
| VCReg | `enc_resblock_6` | 0.637907 | 0.208055 | 0.566421 | 0.358366 |

逐图结果显示，两个模型在全部 304 张图、全部 5 个层位上的 $\Delta_{source}$ 均为正。因此，raw Encoder 中的基本现象很稳定：$G(A)$ 虽然经过生成器解码，但重新编码后仍然比 $B$ 更接近 $A$。

## 5. CUT 与 VCReg 的差异

VCReg 相对 CUT 的 source preference 变化为：

| 层 | VCReg - CUT 均值 | VCReg 更低的逐图比例 |
| --- | ---: | ---: |
| `stem` | -0.027719 | 88.49% |
| `downsample_1` | -0.009263 | 76.32% |
| `downsample_2` | +0.004929 | 39.47% |
| `enc_resblock_1` | -0.054109 | 93.09% |
| `enc_resblock_6` | -0.088828 | 98.03% |

主要观察：

1. VCReg 没有消除源域偏好。最深层仍有 $\Delta_{source}=0.358366$，说明 $G(A)$ 在 raw Encoder 中仍明显更接近 $A$ 而不是 $B$。
2. VCReg 的变化在深层最明显。`enc_resblock_6` 的 source preference 从 0.447194 降至 0.358366，绝对下降 0.088828，约下降 19.86%。
3. `enc_resblock_1` 也有一致下降；`downsample_2` 则略微上升，说明 VCReg 不是对所有层施加同方向的简单收缩。
4. $d(A,B)$ 在深层从 CUT 的 0.682197 降到 VCReg 的 0.637907，而 $d(A,G(A))$ 从 0.177999 升到 0.208055，$d(B,G(A))$ 从 0.625193 降到 0.566421。source preference 的下降主要来自生成结果相对 $B$ 更接近，同时生成结果相对 $A$ 的距离有所增加。

## 6. 与 Expander/NCE 空间的关系

历史 NCE 诊断在同样的五个目标层上先经过 checkpoint 中保存的 `PatchSampleF`/Expander，再计算归一化 embedding 的 cosine distance。raw Encoder 结果则直接使用 Encoder 通道向量。两者的数值不能直接等同，但可以比较关系方向。

历史 NCE 空间的 source preference 均值为：

| 模型 | `stem` | `downsample_1` | `downsample_2` | `enc_resblock_1` | `enc_resblock_6` |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUT | 0.624791 | 0.701417 | 0.716439 | 0.726204 | 0.634453 |
| VCReg | 0.603624 | 0.666050 | 0.684736 | 0.686066 | 0.599544 |

raw Encoder 与 NCE 空间给出一致的定性结论：

- 两个模型的 $G(A)$ 都更接近 $A$，而不是 $B$；
- VCReg 的 source preference 整体低于 CUT；
- VCReg 的下降在深层 raw Encoder 中尤其明显；
- 不能把 raw Encoder 的绝对距离与 NCE embedding 距离直接比较，因为 Expander 改变了特征坐标和归一化后的几何结构。

因此，目前更稳妥的解释是：VCReg 的域对齐迹象并不只是在 Expander 投影后才出现，至少部分已经反映在 Encoder 的深层表示中；但 raw Encoder 仍保留明显源域信息，不能据此声称已经完成域不变表示学习。

## 7. 输出文件

- 分析脚本：[analyze_encoder_domain_alignment.py](../analyze_encoder_domain_alignment.py)
- 全量配置：[manifest.json](../results/feature_analysis/encoder_domain_alignment/full/manifest.json)
- 逐图指标：[per_image_metrics.jsonl](../results/feature_analysis/encoder_domain_alignment/full/per_image_metrics.jsonl)
- 汇总 JSON：[summary.json](../results/feature_analysis/encoder_domain_alignment/full/summary.json)
- 汇总表：[summary.md](../results/feature_analysis/encoder_domain_alignment/full/summary.md)

## 8. 下一步

下一步应继续区分三种因素：

1. `A/B` 域间距本身是否因 VCReg 改变；
2. `A/G(A)` 是否因生成器输出更接近清晰域而增大；
3. `B/G(A)` 是否下降得更多，从而造成 source preference 变小。

可在逐图结果上加入 bootstrap 置信区间和配对 Wilcoxon 检验，再按完整 Encoder 层序列绘制变化曲线。之后再结合 Expander 前后同一图像的局部 patch 对应关系，分析域对齐主要发生在 Encoder 哪一段以及是否由投影层进一步放大或压缩。