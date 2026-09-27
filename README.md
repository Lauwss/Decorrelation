# Decorrelation：非配对图像迁移中的跨域表征几何研究

本项目研究如何在 CUT（Contrastive Unpaired Translation）非配对图像迁移框架中，通过 VCReg 风格的 variance-covariance regularization 改善跨域表征的稳定性。

当前任务以 RRSHID 去雾为主要实验场景。研究重点已经从“是否能降低通道相关性”推进到一个更具体的问题：**PatchNCE 为什么会使生成结果保留源域偏好，以及 Expander 上的 VCReg 是否能缩小这种跨域几何缺口。**

> 训练严格使用非配对的 `trainA/trainB` 图像；配对的 `testA/testB` 关系只用于离线诊断，不参与训练损失。

## 核心问题

CUT 中的 PatchNCE 通过同位置 patch 对比保持输入内容，但它主要把生成结果拉向输入域 A 的局部表征。GAN 则匹配目标域 B 的边际分布，却不保证每个 `G(A)` 与对应的 `B` 在特征空间中逐样本对齐。

本项目用以下量诊断这种源域锚定：

$$
\Delta_{source}=d(B,G(A))-d(A,G(A)),
$$

其中 NCE embedding 的距离为归一化余弦距离：

$$
 d(z_1,z_2)=1-\hat z_1^\top \hat z_2.
$$

`Delta_source` 越大，表示生成结果相对更接近输入雾域，而不是配对清晰域。

## 方法

模型沿用 CUT 的 Encoder-Decoder、PatchGAN 和 PatchNCE。选定的 Encoder feature map 经 `PatchSampleF` 投影到 256 维 Expander 空间：

```text
Encoder feature map
    -> patch sampling
    -> Linear(C, 256) -> ReLU -> Linear(256, 256)
    -> projected feature
    -> L2 normalization
    -> PatchNCE embedding
```

VCReg 作用在未归一化的 `projected` feature 上：

- covariance 项抑制 batch 内通道间的线性冗余；
- variance 项防止表示整体收缩或坍塌；
- PatchNCE 继续承担局部内容保持；
- GAN 继续承担目标域边际分布匹配。

NCE/Expander 是本项目的**主分析空间**，因为 PatchNCE 与 VCReg 的训练约束实际作用于这里。raw Encoder 分析只作为 additional evidence，不能与 NCE 空间的绝对距离直接比较。

## 当前最重要的发现

在 304 对 `testA/testB` 图像上，`enc_resblock_6` 的 NCE 主分析结果为：

| 模型 | `d(A,G(A))` | `d(B,G(A))` | `Delta_source` |
| --- | ---: | ---: | ---: |
| CUT | 0.155784 | 0.790237 | 0.634453 |
| VCReg | 0.130077 | 0.729620 | 0.599544 |

这说明：

1. 标准 CUT 的生成结果在 NCE 空间中存在明显源域偏好；
2. VCReg 后 `d(B,G(A))` 和 `Delta_source` 下降，跨域逐样本几何缺口有所收窄；
3. VCReg 后缺口仍然明显为正，因此当前证据支持“改善跨域对齐”，但还不能宣称已经实现完全域不变；域不变性的范围仍需通过严格消融和跨退化测试确认。

多种子长程实验还显示，Exp4 相比 Exp1 的平均结果为：

- best PSNR：`21.1987 ± 0.1251 dB`；
- final PSNR：`21.1390 ± 0.1048 dB`；
- mean absolute correlation：`0.1036 ± 0.0011`；
- effective rank：`53.5 ± 0.8`。

这些结果说明去相关并非简单地把表示压缩到低秩空间，但 covariance-only、variance-only 和严格匹配的 full 消融仍需完成。

## 已完成工作

- CUT 风格的模块化 Encoder、Decoder、PatchGAN 和 PatchNCE 训练管线；
- Expander projected feature 上的 variance/covariance regularization；
- baseline CUT 与 VCReg 多种子长程训练；
- ASM 和常见退化模块的实现与初步测试；
- NCE 空间中 A/B/`G(A)` 的逐图、逐层静态诊断；
- raw Encoder 中的域对齐辅助诊断；
- NCE 主分析和 raw Encoder 辅助证据的可视化报告；
- 研究需求、实验计划、完成情况和项目架构文档。

## 当前限制与下一步

当前仍需验证：

- VCReg 是否在严格公平消融中稳定优于 CUT；
- covariance 项单独负责压缩退化方向；
- `Delta_source` 已经随雾浓度建立稳定关系；
- 表征已经跨去雨、去模糊等退化迁移；
- 域探针、检索和协方差谱指标是否与 NCE 距离变化一致。

下一阶段优先级：

1. 完成 no-VCReg、covariance-only、variance-only、full 的严格多种子消融；
2. 增加同域参照、paired retrieval、完整测试集置信区间和训练过程 `Delta` 曲线；
3. 按雾浓度或能见度分析 `Delta_source`，并比较协方差谱、PC1 ratio 和 effective rank；
4. 在机制证据稳定后，评估跨去雨、去模糊等任务的表征迁移和跨退化稳定性。

## 目录结构

```text
config.py                              默认训练配置
train.py                               训练入口
model/                                 Encoder、Decoder、Generator、Expander
utils/loss.py                          GAN、PatchNCE、VCReg 损失
analyze_features.py                    NCE/Expander 主诊断
analyze_encoder_domain_alignment.py    raw Encoder 辅助诊断
visualize_encoder_domain_alignment.py  NCE 与 raw 结果可视化
data/                                  非配对训练数据接口
datasets/RRSHID/                       RRSHID 数据集
results/ckpt/                          checkpoint 输出目录
docs/                                  研究、实验和架构文档
```

## 复现实验

以下命令在仓库根目录执行，默认使用 `dehaze` conda 环境和 CUDA。具体参数、checkpoint 名称及数据路径以脚本帮助和配置文件为准。

训练：

```bash
conda activate dehaze
python train.py --help
python train.py
```

运行 NCE/Expander 主诊断：

```bash
python analyze_features.py --help
python analyze_features.py
```

运行 raw Encoder 辅助域对齐分析：

```bash
python analyze_encoder_domain_alignment.py --help
python analyze_encoder_domain_alignment.py
```

生成可视化报告：

```bash
python visualize_encoder_domain_alignment.py --help
python visualize_encoder_domain_alignment.py
```

分析结果和图表默认位于 `results/feature_analysis/`，训练 checkpoint 默认位于 `results/ckpt/`。由于 checkpoint 和实验输出通常较大，仓库只保留代码、配置和研究文档，不把生成结果当作源代码提交。

## 文档索引

- [VCReg 跨域不变表征研究项目文档](docs/VCReg_去雾解耦研究项目文档.md)：研究问题、机制假说、实验设计和当前主线。
- [VCReg 跨域不变表征实验完成情况报告](docs/VCReg_去雾解耦实验完成情况报告.md)：已完成结果、证据边界和下一阶段实验清单。
- [Encoder 空间域对齐分析报告](docs/Encoder空间域对齐分析报告.md)：NCE 主分析与 raw Encoder 辅助证据。
- [项目架构与工作总览](docs/项目架构与工作总览.md)：代码职责、梯度边界和复现注意事项。
- [可视化报告](results/feature_analysis/encoder_domain_alignment/visual_report/report.md)：生成后可查看 NCE 和 raw Encoder 图表。

## 研究定位

当前项目最准确的定位是：**非配对图像迁移中的跨域表征几何诊断与 VCReg 去相关正则化。**

研究目标是通过可测量的 NCE 空间指标，理解并逐步减弱非配对 CUT 的源域锚定，验证 VCReg 是否能在保持内容和生成质量的同时学习更好的跨域不变表征。
