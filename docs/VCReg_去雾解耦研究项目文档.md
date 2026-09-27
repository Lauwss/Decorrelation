# 基于 VCReg 的无监督去雾：跨域不变表征与紧凑嵌入空间研究项目文档

> 版本：v1.2 ｜ 日期：2026-09-26
> 状态：阶段性研究文档（已完成 NCE 主诊断、VCReg 长程原型和 raw Encoder 辅助分析；严格机制闭环仍在进行）
> 关键词：Unpaired Image Dehazing, CUT, PatchNCE, VICReg/VCReg, Decorrelation, Minimal-Guidance Disentanglement, Domain Transfer, Representation Learning

---

## 1. 项目概述

### 1.1 一句话定位

**在 unpaired domain transfer（CUT 范式）的 fine-tuning 阶段，将 variance-covariance regularization（VCReg）施加到 PatchNCE 使用的 Expander feature 上，抑制域特异的相关冗余，缩小跨域逐样本几何缺口，进而学习更接近跨域不变的表征。**

当前研究不把“通道去相关”直接等同于完整 disentanglement。项目的可证伪目标是：在不破坏内容保持和生成质量的前提下，减少生成结果在 NCE 空间中对输入域的过度依赖，使 $G(A)$ 更接近配对清晰域 $B$，并在后续跨退化迁移实验中体现出更好的表征稳定性。

### 1.2 核心判断（立项依据）

1. **已确认的主现象**：在训练实际使用的 PatchNCE/Expander 空间中，CUT 的 $d(A,G(A))$ 明显小于 $d(B,G(A))$，说明生成结果容易停留在输入域表征附近，存在可测量的跨域逐样本对齐缺口。
2. **VCReg 的当前证据**：VCReg 长程原型降低 Expander 通道相关性，并在完整配对测试集上降低 $d(B,G(A))$ 和 source preference；这支持“去相关正则有助于减弱源域锚定”的判断，但尚不足以证明完成了通用的 degradation/content disentanglement。
3. **辅助证据**：raw Encoder 也观察到同方向的域关系变化，说明部分变化可能在进入 Expander 前已经发生；但模型优劣和主要结论仍以 NCE/Expander 空间为准，因为 PatchNCE 与 VCReg 的训练约束作用于该空间。
3. **文献空白**：VCReg 类方法目前**只用于 pretraining 语境**（SSL / supervised pretraining），从未在 domain transfer / restoration 的 fine-tuning 阶段使用，更从未被当作解耦工具。
4. **动机同构**：VCReg 论文防 shortcut 的初衷 ≈ disentanglement 的初衷；但 VCReg 给出的是**几何/能量意义上的独立性**（协方差结构），与 VAE 系**概率意义上的独立性**（对角后验）是不同的解耦语义，且 VCReg 可直接挂在 CUT encoder 上，无需重构分支与重参数化。
5. **场景独有的弱引导**：大气散射模型只有 (β, A) 两个物理参数，清晰图 + 随机参数可批量合成 hazy 图，天然构造"只差退化因子"的样本对——这正满足 weakly-supervised disentanglement 的 pairs 设定，且**无需任何人工标注**。
6. **定位决策**：叙事重心放在 restoration / 低层视觉社区，representation quality 作为差异化证据出现（跨退化迁移 + 雾天下游任务），不正面进军 SSL 赛道、不在 ImageNet 刷榜。

### 1.3 方法学设定优势（v1.1 新增）

**训练非配对，分析用配对（unpaired training + paired diagnostics）**：使用 RESIDE 类合成数据集，训练时模型从未接触配对信号（严格 unpaired 设定），评估与分析时利用其配对关系做精确测量。这使跨域对齐的测量比纯 unpaired 工作（只能做检索近似）严格一个档次，且不削弱非配对设定声明——**配对仅用于评估**。

### 1.4 明确不做的事

- 不提出"新 encoder 架构"（贡献是训练范式/正则方案，措辞用 framework / paradigm / regularization scheme）。
- 不做纯理论文章（理论分析作为论文的 analysis section）。
- 不做"disentanglement-based pretraining 新范式"的正面对决（这是 Paper 2 的候选，由实验结果投票决定）。
- 不与 ODCR（CVPR 2025, orthogonal decoupling contrastive regularization for unpaired dehazing）撞车——需在 related work 中明确区分（见 §4.6）。

### 1.5 当前研究问题

本项目当前围绕三个递进问题展开：

1. **诊断问题**：为什么非配对 CUT 在 PatchNCE 空间中容易保持输入域特征，而不是实现逐样本的目标域对齐？
2. **机制问题**：VCReg 的 covariance 与 variance 约束是否通过压缩相关的域特异方向，降低 $\Delta_{source}=d(B,G(A))-d(A,G(A))$？
3. **表征问题**：这种 NCE 空间中的改善能否转化为对退化变化更稳定、可迁移到其他复原任务的 Encoder 表征？

当前已回答第一个问题，并得到第二个问题的初步支持；第三个问题尚未完成验证。

---

## 2. 背景与问题定义

### 2.1 任务设定

- **任务**：单图像去雾（single image dehazing），unpaired 训练设定——有清晰图集 {x_c ~ P_clear} 与有雾图集 {x_h ~ P_hazy}，**训练时无逐像素配对**；评估与分析使用配对关系（见 §1.3）。
- **基线**：CUT（Contrastive Unpaired Translation, Park et al., ICCV 2020），PatchNCE 损失 + GAN。
- **物理模型**：大气散射模型 x_h = t(x_c) · x_c + A · (1 − t(x_c))，t = e^(−β·d)。

### 2.2 已知问题（CUT 的病灶）

- PatchNCE 存在 **shortcut learning** 与训练不稳定问题（后续工作 Cut&Tag、FastCUT 均在缓解），其本质是 encoder 过度依赖少数主导线索（低频次雾/纹理 shortcut）——即 **gradient starvation** 与表征坍缩倾向。

### 2.3 关键诊断观察与"假对齐"现象（v1.1 核心节）

**观察**（配对分析，A = 雾域，B = 清晰域，G(A) = 去雾输出）：

| 量 | baseline CUT | +VCReg | 含义 |
|---|---|---|---|
| cos d(A, G(A)) | [0.10, 0.20] | 略微下降 | 域内对齐（PatchNCE 直接约束） |
| cos d(B, G(A)) | [0.70, 0.80] | 下降（不多） | 跨域逐样本对齐 |
| **Δ = d(B, G(A)) − d(A, G(A))** | 大 | **显著减小** | **跨域对齐缺口（headline 指标）** |

**解读——"假对齐"（false alignment）**：d(A, G(A)) 小而 d(B, G(A)) 大，说明 G(A) 并未进入清晰域流形，只是落在 **A 域流形内 E(A) 的附近**。encoder 的表征空间是**域分层（domain-stratified）**的：A、B 两域各居一个流形，流形间存在系统性间隔。PatchNCE 只约束"输出对 encoder 而言像输入"，而 encoder 拥有大量域特异特征可用——G(A) 以最小代价（在 A 流形内部挪一小步）满足损失。这是 CUT shortcut learning 的 embedding 空间版本。

**理论上完全可预期**：PatchNCE 的锚永远在 A 侧，GAN 损失只匹配 B 的边际分布、不匹配逐样本——跨域逐样本对齐**从来不在任何一项损失的优化视野内**。观察到 [0.7, 0.8] 不是 bug，而是目标函数设计的直接后果。**VCReg 是唯一把跨域几何也纳入优化的项**，这使其作用解释（§3）与论文动机天然闭环。

**形式化**：设 z = μ_dom + Wc + W_d·δ（μ_dom 为域相关均值偏移/主方向，c 为 content，δ 为退化）。PatchNCE 最小化 d(z_A, z_G(A)) 时，μ_A 与 W_dδ_A 分量为 A 域内常量、梯度不可见——**损失的天花板位于 A 流形内部，跨域间隔完全不被优化**。

### 2.4 当前阶段的证据总览

当前最重要的分析在 304 对 `testA/testB` 图像上完成，并以训练实际使用的 PatchNCE/Expander 空间为主。对每个模型分别计算：

$$
d(A,B),\quad d(A,G(A)),\quad d(B,G(A)),\quad
\Delta_{source}=d(B,G(A))-d(A,G(A)).
$$

`enc_resblock_6` 的全量均值如下：

| 模型 | $d(A,B)$ | $d(A,G(A))$ | $d(B,G(A))$ | $\Delta_{source}$ |
| --- | ---: | ---: | ---: | ---: |
| CUT | 0.794340 | 0.155784 | 0.790237 | 0.634453 |
| VCReg | 0.730024 | 0.130077 | 0.729620 | 0.599544 |

这组结果支持三点：

1. CUT 和 VCReg 都存在源域偏好，$G(A)$ 与 $A$ 的 NCE 距离远小于与 $B$ 的距离；
2. VCReg 使最深层的 source preference 下降约 5.5%，同时 $d(B,G(A))$ 下降，说明清晰域侧的几何缺口有所收窄；
3. 这不是“已经实现域不变性”的证明，因为 VCReg 后 $\Delta_{source}$ 仍明显为正，且 $d(A,G(A))$ 也发生变化。

除 NCE 主证据外，raw Encoder 的 `enc_resblock_6` source preference 从 CUT 的 0.447194 降至 VCReg 的 0.358366。该结果作为 additional evidence，说明部分域关系变化可能在 Expander 之前已经出现；raw Encoder 的绝对距离不与 NCE 距离直接比较。

### 2.5 VCReg 如何可能改善跨域不变表征

当前最稳妥的机制解释是一个“几何缺口收窄”路径，而不是直接宣称显式解耦：

1. **PatchNCE 提供内容保持，但锚点偏向 A 域。** 它鼓励 $G(A)$ 在同位置保持与 $A$ 相似的局部结构，因此天然容易保留雾域特征。
2. **GAN 只约束目标域边际分布。** 它要求生成集合看起来像 B 域，但不要求每个 $G(A)$ 与对应的 $B$ 在表征空间逐样本对齐。
3. **Expander 中的相关通道可能承载域特异 shortcut。** 雾造成的低频、低对比度和纹理变化可能以跨通道相关方向进入 projected feature，使输入域与清晰域形成稳定的几何偏移。
4. **VCReg 的 covariance 项压缩冗余相关方向，variance 项防止整体收缩。** 在 PatchNCE 仍负责局部内容保持的前提下，VCReg 可能减少依赖单一域特异方向的自由度，使生成结果从 A 域流形内部的“最小移动”转向更接近 B 域的表示区域。
5. **结果表现为 source preference 下降。** 这意味着生成结果与 B 的距离相对下降，或与 A 的距离相对上升，最终缩小 $\Delta_{source}$；这是目前对“更好的跨域不变性”的操作性定义。

这里的“不变性”是功能性的、相对于域变化的几何稳定性，不等于数学意义上的完全域不变，也不等于已经分离出可独立交换的 content/degradation latent。

### 2.6 当前工作进展

| 工作方向 | 状态 | 当前结论 |
| --- | --- | --- |
| CUT baseline 与训练基础设施 | 已完成 | 有固定 checkpoint、测试和特征诊断流程 |
| Expander VCReg 接入 | 已完成 | covariance/variance 正则已接入 PatchNCE 使用的 projected feature |
| VCReg 长程多种子原型 | 已完成 | Exp4 相比 Exp1 平均 PSNR 和通道相关性均有改善 |
| NCE 空间 $A/B/G(A)$ 诊断 | 已完成静态全量版 | 304 对图像、两模型、五层，主分析结果已生成可视化报告 |
| raw Encoder 对照 | 已完成辅助版 | 支持部分变化在 Expander 前出现，但不作为主结论 |
| ASM 与常见退化模块 | 工程实现完成 | 第一轮稳定性实验已完成，但不替代 NCE 域对齐结论 |
| 严格 var-only/cov-only/full 消融 | 未完成 | 当前不能解释两项正则的独立分工 |
| 训练过程 $\Delta$ 曲线 | 未完成 | 当前只有 checkpoint 结束后的静态测量 |
| 雾浓度与 $\Delta$ 关系 | 未完成 | 尚无按 $\beta$/能见度分桶的主实验 |
| 同域参照、检索和域探针 | 未完成 | 尚不能完全排除内容差异或量化域可分性变化 |
| 显式 latent swap、DCI/MIG | 未完成 | 尚不能宣称完整 disentanglement |
| 跨退化迁移 | 未完成 | 尚不能宣称通用 degradation-invariant transfer |

因此，项目已经从“VCReg 是否有效”的原型阶段进入“VCReg 为什么有效、改善是否可泛化”的验证阶段。

---

## 3. 机制假说（核心假说，可证伪）

> **H1（退化主导方向假说，v1.1 细化）**：雾是一种高跨通道相关性的退化——降对比度、抹平纹理，使特征各通道被同一个平滑的"雾方向"主导。该方向在特征协方差矩阵中表现为巨大的高相关成分，正是它撑开了 A、B 两流形的间隔。VCReg 的 covariance 项惩罚 off-diagonal 能量，**直接攻击这个最大相关方向**；variance hinge 防止压缩释放的能量预算被坍缩吃掉。因此 VCReg 不是平均地拉近一切，而是**对退化诱导的相关结构有针对性**。

> **H2（分工假说）**：VCReg 三要素在 CUT 场景中分工明确——covariance 项负责压缩退化相关主导方向，variance 项负责防止全局坍缩，PatchNCE 扮演 invariance 角色负责 A 域内正样本对齐。CUT + VCReg 实际上是 VICReg 三件套的一个"意外完整"实例。

**余弦距离分解**（线性模型下）：

设 z = Wc + W_d·δ（配对设定下 c 跨域共享，δ 为退化增量），则

    Δ ≈ 1 − cos(z_B, z_G(A)) − [1 − cos(z_A, z_G(A))]

跨域缺口 Δ 由域均值偏移项 μ_A − μ_B 与退化残差项 W_dδ 在 content 子空间的投影共同主导（analysis section 中把这两项从 1−cos 界中单独拎出，见 §6.1）。

**可证伪预测**：

- **P1（分工）**：cov-only 与 var-only 的 ablation 应呈系统性差异——cov-only 压缩域间隔但可能牺牲稳定性；var-only 改善坍缩但不产生正交化收益。
- **P2（浓度依赖，v1.1 新增）**：Δ 随雾浓度（β / 能见度）增大而增大（雾方向能量大）；**VCReg 在高雾区的 Δ 改善幅度应最大**。合成数据可精确控制 β，此预测可严格检验。
- **P3（域分层，v1.1 新增）**：baseline CUT 的 embedding 上，线性域探针（区分 A/B）准确率应显著高于随机，且 G(A) 多被分类为 A 域；加 VCReg 后域可分性下降、G(A) 被判为 B 域的比例上升。
- **P4（谱结构，v1.1 新增）**：baseline 中 A 域 embedding 谱比 B 域更"平"（前几个主成分方差集中，有效秩低）；VCReg 使 A 域谱有效秩上升、与 B 域谱形靠近。

---

## 4. 相关工作

### 4.1 VICReg / VCReg 谱系

- VICReg（Bardes et al., ICLR 2022）：variance hinge + invariance + covariance penalty 三件套；covariance penalty 与 Barlow Twins 的 cross-correlation 目标有内在联系。
- Mialon et al. (2022)：VCReg 通过随机 MLP projector 反向传播，在 encoder 输出上强制 pairwise independence；covariance 惩罚等价于最小化各维度间 HSIC 之和——**支撑 decorrelation → 某种解耦的假设**。
- Shwartz-Ziv et al. (2023, 信息论分析)：variance/covariance 项 ≈ 最大化表征熵的 proxy；invariance 项 ≈ 最小化条件熵；**仅 covariance 惩罚会导致坍缩，必须与 invariance/variance 平衡**——支撑 H2。
- Zhu et al. (ICML 2024, "Variance-Covariance Regularization")：VCReg 用于中间层，显式讨论 gradient starvation 与 neural collapse，CDNV 度量；在 long-tail、hierarchical classification 的预训练与迁移上 SOTA。
- 2025 年延伸工作（kernel VICReg、spectral VICReg、SAG-VICReg 等）：全部停留在 SSL/预训练语境——**确认文献空白**。

### 4.2 CUT 与 unpaired translation

- CUT（ICCV 2020）：PatchNCE。
- FastCUT / Cut&Tag：效率与 shortcut 缓解。
- 去雾领域的 CUT 变体（多篇）；**ODCR（CVPR 2025）用词与我们的方向高度重合（orthogonal + decoupling + contrastive + unpaired dehazing），必须精读并明确区分**：其大概率为 feature-level orthogonal contrastive loss，而非协方差结构正则 + 弱引导解耦。

### 4.3 Disentanglement 理论

- Locatello et al. (ICML 2019)：无监督解耦不可辨识；随机种子与超参影响大于模型选择——**因此我们的分解假设必须是功能性的（z = μ_dom + Wc + W_dδ），而非结构性的（不做显式 z_c/z_d 切块）**。
- Shu et al. (ICLR 2020)：弱监督解耦 guarantee 框架（restricted labeling / match pairing / rank pairing）。
- Locatello et al. (2020, "Weakly-Supervised Disentanglement Without Compromises", Ada-GVAE)：成对观测（只差少数 factor）即可可靠解耦，无需标注哪个 factor 变了。
- AAAI 2020 pairwise similarity：样本对相似度标签显著改善解耦。

### 4.4 与 JEPA 的关系（仅作叙事参照，不对标）

JEPA 在表征空间做预测、避免像素级重构。我们的方法同为 embedding 空间的训练信号（PatchNCE + 二阶统计约束），但没有预测器架构，也无意对标 JEPA 量级的预训练评价协议。仅在 related work / discussion 中提一句"embedding-space training signal for restoration"的定位。

---

## 5. 方法设计

### 5.1 总体框架：VCReg-augmented CUT + Physics-Guided Weak Supervision

在 CUT 的 encoder E（即生成器 G 的 encoder）的瓶颈特征 z = E(x) 上，施加 VCReg 两项（λ_v, λ_c 可调），并可选引入物理引导的解耦监督。

**损失函数**：

    L_total = L_GAN + λ_NCE · L_PatchNCE + λ_v · L_variance + λ_c · L_covariance + λ_g · L_guide

- **L_variance**：hinge 形式，max(0, 1 − sqrt(Var(z_i)))，对逐通道方差施加下界。
- **L_covariance**：off-diagonal covariance 的平方和 C_ij²（i ≠ j）。
- **L_guide（弱引导，可选/可调权重，v1.1 注意：§2.3 发现后此项优先级下调——见 §5.3）**：利用散射模型合成对 (x_c, x_h = S(x_c; β, A))。两种候选形式：
  - (a) *Factor-consistency*：E(x_h) 与 E(x_c) 在 content 子空间内对齐（借鉴 Ada-GVAE 的 pairs 思想）；
  - (b) *Degradation-swap / contrastive*：同 content 不同 (β, A) 的 hazy 样本，其表征差应落于由退化参数张成的子空间。
  - 注意：引导只用**合成数据**，真实 unpaired 数据仍走 CUT 主损失——合成数据不破坏 unpaired 设定。

### 5.2 实现细节（待定项）

- VCReg 施加位置：瓶颈层 vs. 中间多层（ICML 2024 论文表明中间层更有效，待验证）。
- projector：VICReg 惯例加随机 MLP projector 再接 VCReg，encoder 主干接 PatchNCE；是否共享/分离 projector 待实验。
- batch size：covariance 估计对 batch size 敏感，小 batch 下考虑 running covariance 或 Barlow Twins 式折中。
- λ 初值：λ_v = λ_c = 1（VICReg 惯例），λ_g 从小往大扫。
- **测量协议（v1.1 新增）**：d 统一在瓶颈层、eval 模式下测量，同时报告中间层版本；确认无 stop-gradient 等干扰。

### 5.3 研究路线优先级调整（v1.1）

§2.3 的"假对齐"发现改变了任务分解：

1. **第一优先：Δ 缺口的机制与消除**——VCReg 不加 L_guide 时能收窄 Δ 多少？这是零额外监督（保持纯 unpaired）即可讲的核心故事。
2. **第二优先：解耦与弱引导**——L_guide 作为增量贡献，回答"最少需要多少引导"。
3. 若 VCReg 单独已显著收窄 Δ 且解耦证据充分，论文甚至可以纯 unpaired + 物理先验（散射模型仅用于分析与合成评估数据），故事更干净。

---

## 6. 理论分析计划（Analysis Section）

### 6.1 线性模型命题（草案，v1.1 围绕 Δ 展开）

**设定**：z = μ_dom + Wc + W_dδ；c, δ 零均值、协方差已知；W 线性 encoder；配对设定下两域共享 c；退化图为 x' = x + δ（散射模型一阶展开）。
**VCReg 理想化**：约束 off-diagonal(W Σ_x Wᵀ) = 0，且 diag ≥ γ。
**命题（目标形态）**：把 1 − cos 的 Cauchy-Schwarz 界展开后，Δ 的主导项为域偏移投影 ‖P_c(μ_A − μ_B)‖² 与退化残差投影 ‖P_c W_dδ‖²（P_c 为 content 子空间投影）；covariance 约束使退化相关方向（高 off-diagonal 能量方向）的正交化以 O(‖off-diag C‖_F) 的速度降低上述投影。
**证明路线**：cos 展开 → Cauchy-Schwarz 间隙由正交分量主导 → Lagrangian 分析能量预算约束下 off-diagonal 收缩对域相关方向的作用。

### 6.2 Toy 验证实验

- 2 维/低维线性 encoder + 合成退化（已知 ground-truth μ_dom, W_c, W_d），closed-form 计算 Δ 随 λ_c 的变化曲线，对照命题预测。
- 非线性版本（2 层 MLP）做 sanity check。

### 6.3 表征结构度量

- **CDNV**（Class-Distance Normalized Variance）、within-class covariance 结构、embedding 各向异性（特征谱 / effective rank）。
- **Δ 及其分解**（v1.1 新增，见 §7.5）。

---

## 7. 实验设计

### 7.1 主实验（去雾性能）

- **数据集**：RESIDE（ITS/OTS/SOTS-outdoor）、NH-HAZE、D-HAZE；合成引导与诊断数据由清晰图 + 可控 (β, A) 生成。
- **指标**：PSNR、SSIM、LPIPS、FID；无参考指标（NIQE）作辅证。
- **对比**：CUT、FastCUT、Cut&Tag、有监督 SOTA（引用上限）、ODCR（必读、必比）、其他 CUT-based dehazing 变体。

### 7.2 消融实验（对应机制假说）

| 编号 | 实验 | 验证内容 |
|---|---|---|
| A1 | var-only / cov-only / var+cov | H2 / P1：三要素分工 |
| A2 | λ_v, λ_c, λ_g 敏感性 | 稳定性与最佳配比 |
| A3 | VCReg 施加层（瓶颈 vs 中间层） | ICML 2024 结论是否迁移到我们的 setting |
| A4 | 无 L_guide vs 两种 L_guide | 最小引导究竟多小够用 |
| A5（新） | Δ、域探针准确率、有效秩随 λ_c 变化 | H1/P3/P4 的机制指标 |

### 7.3 解耦验证实验（对应 RQ2）

- **退化可控性**：固定 content，调节 z 中退化方向分量，生成不同浓度/大气光的雾图（可视化 + 与真实散射参数的相关性定量分析）。
- **退化交换**：同 content 不同 (β, A) 样本间交换退化分量，观察 content 保持、雾型变化。
- **解耦度量**：引导充分时报告 DCI / MIG（合成数据 ground-truth factor 已知）。

### 7.4 表征迁移实验（对应 RQ3，Paper 1 的"超额证据"）

- **跨退化迁移**：去雾训好的 encoder **freeze 后接** 去雨（Rain100L/Rain800）/ 去模糊（GoPro），fine-tune 对比：(i) 随机初始化，(ii) SimCLR/VICReg ImageNet 预训练，(iii) 无 VCReg 的 CUT encoder。**这是决定 Paper 2 是否立项的关键实验。**
- **下游任务**：Foggy Cityscapes / RTTS 上的雾天检测（检测器 fine-tune），对比 baseline encoder。
- **低成本决策原则**：该实验应尽早（1–2 周原型）完成，用结果决定叙事收敛点。

### 7.5 embedding 空间与 Δ 缺口分析（v1.1 大幅扩充）

**防御性标定（排除平凡解释）**：
- 参照距离：d(A, A′)、d(B, B′)（同域跨场景）——证明 [0.7, 0.8] 是跨域间隔而非内容差异。
- 检索准确率：E(G(A)) 的最近邻是否为其配对 E(B)（也作为 unpaired 声明的佐证）。

**机制验证**：
- **Δ 训练曲线**（CUT vs +VCReg）：渐近值、收敛速度。
- **Δ–浓度曲线**（P2）：Δ 随 β / 能见度变化；VCReg 改善幅度是否在高雾区最大。预期为论文核心图之一。
- **域可分性探针**（P3）：冻结 E 上训练线性 A/B 分类器；报告 G(A) 被判为 B 域的比例。
- **PCA 谱分析**（P4）：A / B / G(A) 各自的特征谱、有效秩、各向异性比；检验"雾方向 = 高相关主导成分"。

---

## 8. 论文故事线（Paper 1）

**标题方向**：*Closing the Cross-Domain Alignment Gap in Unpaired Dehazing with Variance-Covariance Regularization*（待定）

**贡献列表（按审稿价值排序，v1.1 更新）**：

1. **发现与诊断**：揭示 CUT 在无配对去雾中的"假对齐"现象——域内对齐良好（d(A, G(A)) 小）但跨域逐样本对齐存在巨大缺口（d(B, G(A)) 大），并给出其目标函数层面的解释（跨域对齐不在 PatchNCE/GAN 任一损失的优化视野内）；提出 Δ = d(B, G(A)) − d(A, G(A)) 作为跨域对齐缺口的诊断指标。
2. **首次将 VCReg 引入 domain transfer / restoration 的 fine-tuning 阶段**（跳出 pretraining 语境），显著收窄 Δ，并系统验证其对表征结构的改善（域可分性、有效秩、CDNV）。
3. **机制分析**：提出"退化诱导的高相关主导方向被 covariance 项针对性压缩"的解释（H1），给出线性模型下围绕 Δ 的余弦距离界（§6.1），并以 toy 实验 + 浓度依赖曲线（P2）+ 谱分析（P4）佐证。
4. **物理引导的最小监督解耦（增量贡献）**：散射模型免费构造 pairs，实现 degradation/content 功能性解耦与退化可控生成（几何解耦路径，与 VAE 系概率解耦形成对照）。
5. **跨退化迁移证据**：解耦 encoder 的 degradation-invariance 与迁移能力。

**Related work 必须处理**：ODCR（明确区分：orthogonal contrastive loss ≠ covariance-structure 正则 + 弱引导解耦）；VCReg/ICML 2024（我们用的是其方法、但场景与目的不同）；Ada-GVAE / Shu et al.（我们借其 pairs 思想，但非 VAE、非通用解耦 benchmark）。

---

## 9. 里程碑计划（建议 6 个月）

| 阶段 | 时间 | 交付物 |
|---|---|---|
| M0 准备 | 第 1–2 周 | 精读 ODCR 并写差异分析；复现 CUT 去雾 baseline；VCReg 接入代码跑通；**钉死 d 的测量协议（层、eval 模式、无梯度干扰）** |
| M1 关键决策实验 | 第 3–4 周 | A1 消融初版 + **Δ 曲线与防御性标定**（§7.5）+ **跨退化迁移原型实验**（§7.4）→ 决定叙事收敛点 |
| M2 主实验 | 第 2–3 月 | 主表全部对比、全部消融（含 A5 机制指标）、Δ–浓度曲线 |
| M3 解耦实验 | 第 3–4 月 | §7.3 全部 + L_guide 形式定型（视 M1 结果决定优先级） |
| M4 理论 + 写作 | 第 4–5 月 | Δ 的线性模型命题与 toy 验证；论文初稿 |
| M5 投稿准备 | 第 6 月 | 内审、补充实验、投稿（目标：CVPR/ICCV/ECCV 或 TPAMI 视结果定） |

---

## 10. 风险与备选方案

| 风险 | 概率 | 影响 | 备选方案 |
|---|---|---|---|
| ODCR 与我们方向重叠度高 | 中 | 贡献 4 被稀释 | 精读后切割：主打 Δ 缺口诊断 + covariance 结构正则；解耦降为增量贡献 |
| 跨退化迁移提升微弱 | 中 | Paper 2 取消，贡献 5 缩水 | 叙事收敛为"假对齐诊断 + 缺口收窄 + 机制分析"，删去范式类表述 |
| covariance 估计对 batch size 敏感 | 高 | 训练不稳定 | running covariance / 大 batch / Barlow Twins 式折中 |
| L_guide 引入合成域 bias | 中 | 真实雾域性能下降 | 降低 λ_g；仅对合成 pairs 计算 L_guide；或完全去掉 L_guide（§5.3 路线 1：纯 unpaired + 物理先验） |
| 线性模型理论与深度模型行为不符 | 中 | analysis section 被质疑 | 明确标注适用范围；非线性 toy + 实证相关性曲线补强 |
| 纯去雾性能提升不显著 | 低 | 论文卖点转向表征与诊断 | 主指标增加 LPIPS/FID 与 Δ/域探针等结构指标；强调机制贡献 |
| Δ 改善主要来自 d(A) 也下降（"一起靠拢"而非跨域改善） | 中（v1.1 新增） | H1 被误读 | 同时报告 d(A)、d(B)、Δ 三者；要求 Δ 收窄主要由 d(B) 下降驱动才算机制成立 |

---

## 11. 后续展望（Paper 2，条件触发）

仅当 §7.4 迁移实验结果显著时启动：

**选题方向**：Degradation-Invariant Pretraining——将验证过的解耦机制规模化到多退化类型，提出跨退化预训练范式；评价协议升级为下游任务与跨数据集泛化；此时才正面进入"更好的 representation 提取器"叙事（JEPA 量级工作）。若结果一般，则止步于 Paper 1，损失可控。

---

## 附录 A：术语表

- **VCReg / VICReg**：Variance-Covariance Regularization（variance hinge + covariance penalty）。
- **PatchNCE**：CUT 中的 patch-level contrastive loss。
- **Δ（对齐缺口）**：d(B, G(A)) − d(A, G(A))，跨域逐样本对齐缺口的诊断指标。
- **假对齐（false alignment）**：G(A) 仅落在 A 域流形内部满足 PatchNCE，未进入清晰域流形的现象。
- **CDNV**：Class-Distance Normalized Variance，neural collapse 度量。
- **HSIC**：Hilbert-Schmidt Independence Criterion，covariance 惩罚的上界解释工具。
- **Gradient starvation**：模型过度依赖少数主导特征、其余特征梯度被抑制。
- **ODCR**：CVPR 2025 Orthogonal Decoupling Contrastive Regularization（unpaired dehazing），最相关竞品。

## 附录 B：核心参考文献（待补全引用格式）

1. Bardes et al., VICReg, ICLR 2022.
2. Mialon et al., HSIC analysis of VICReg, 2022.
3. Shwartz-Ziv et al., Information-theoretic analysis of VICReg, 2023.
4. Zhu et al., Variance-Covariance Regularization (intermediate layers, gradient starvation, neural collapse), ICML 2024.
5. Park et al., CUT, ICCV 2020.
6. Locatello et al., Challenging Common Assumptions in Disentanglement, ICML 2019.
7. Shu et al., Weakly Supervised Disentanglement Guarantees, ICLR 2020.
8. Locatello et al., Weakly-Supervised Disentanglement Without Compromises (Ada-GVAE), ICML 2020.
9. Pairwise similarity supervision for disentanglement, AAAI 2020.
10. ODCR, CVPR 2025.
11. Cut&Tag / FastCUT（CUT 后续工作）.
12. RESIDE (Li et al., TIP 2019) 等数据集论文.

## 附录 C：v1.0 → v1.1 变更记录

1. 新增 §1.3 方法学设定优势（unpaired training + paired diagnostics）。
2. 新增 §2.3 关键诊断观察与"假对齐"现象（含数据表与形式化）。
3. H1 细化为"退化主导方向假说"（高跨通道相关性被 covariance 项针对性压缩）；新增可证伪预测 P2（浓度依赖）、P3（域分层探针）、P4（谱结构）。
4. §5.3 新增研究路线优先级调整（VCReg-单独收窄 Δ 为第一优先，L_guide 降为增量贡献）。
5. §6.1 理论命题改为围绕 Δ 展开（域偏移投影 + 退化残差投影）。
6. §7.2 消融表新增 A5（Δ/域探针/有效秩机制指标）；§7.5 大幅扩充为"防御性标定 + 机制验证"两小节（Δ 曲线、Δ–浓度曲线、域探针、PCA 谱分析）。
7. §8 贡献列表更新："假对齐"诊断升为贡献 1，新增 Δ 指标；风险表新增"Δ 改善来源误判"风险。
8. §9 M0/M1 里程碑新增测量协议钉死与 Δ 曲线任务。
