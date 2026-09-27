# VCReg 去雾解耦研究实验完成情况报告

> 日期：2026-09-22  
> 项目：基于 CUT 的非配对去雾与 Expander 去相关  
> 数据集：RRSHID 及其 `hard_sample` 诊断子集  
> 报告性质：基于当前仓库已有代码、checkpoint、训练日志和分析产物的阶段性核查

## 1. 结论摘要

截至 2026-09-22，项目已经完成了“CUT baseline + PatchNCE 源域偏好诊断 + Expander VCReg 风格去相关”的第一阶段实验，但尚未完成项目规划中完整的 disentanglement、跨退化迁移和机制闭环验证。

当前最可靠的研究结论是：

1. **CUT 的 PatchNCE 空间确实存在明显的源域锚定/假对齐现象。** 在困难集、`enc_resblock_6` 层上，`cut_baseline` 的 $d(A,G(A))=0.1728$，而配对清晰参考的 $d(B,G(A))=0.8168$，源域偏好
   $$
   \Delta_{\mathrm{source}}=d(B,G(A))-d(A,G(A))=0.6440.
   $$
   这说明生成结果在训练所使用的 NCE 空间中明显更接近雾输入，而不是配对清晰图。
2. **Expander 去相关能够降低通道相关性，并在长程多种子实验中改善 PSNR。** Exp4（协方差权重 0.1、方差权重 0.25）三种子最终 mean absolute correlation 为 $0.1036\pm0.0011$，Exp1 为 $0.1170\pm0.0184$；最佳测试 PSNR 均值分别为 $21.1987\pm0.1251$ dB 和 $20.8903\pm0.1181$ dB。
3. **去相关实验同时缩短了配对清晰图到生成图的 NCE 距离，但不能据此声称已经实现 content/degradation disentanglement。** 当前结果更直接支持“表示去相关和跨域几何缺口收窄”，尚不足以证明退化因素已经被独立编码或可交换控制。
4. **目前没有完成完整 A1 消融、$\Delta$ 训练曲线、浓度依赖曲线、线性域探针、检索防御性标定或跨退化迁移。** 因此项目文档中的 H1/H2 和 Paper 1 完整贡献列表仍应视为待验证假说，不能全部写成已证实结论。

总体判断：项目已完成 M0 的主要工程基础和 M1 的部分关键诊断，已经具备撰写阶段性论文分析的结果，但距离完整实验闭环还缺少严格消融与外推验证。

## 2. 核查范围与证据规则

本报告只把以下内容视为已完成证据：

- 仓库中实际存在的训练脚本、分析脚本、checkpoint 或 CSV/日志；
- `docs` 中与上述产物一致的实验汇总；
- 能从已有报告明确确认配置、数据集、训练轮数和指标的实验。

计划文档中的命令、未找到 checkpoint 的提交脚本、没有对应指标的目录，不作为“已完成实验”的证据。

主要证据入口：

- [VCReg 研究项目文档](VCReg_去雾解耦研究项目文档.md)
- [Expander 200 epoch 实验报告](expander_decorr_200_epoch_report.md)
- [Expander 去相关实验规范](expander_decorr.md)
- [低成本定位实验计划与结果](低成本定位实验计划.md)
- [项目架构与工作总览](项目架构与工作总览.md)
- [CUT baseline 特征诊断](../results/feature_analysis/cut_baseline/hard_sample/report.md)
- [Exp4 特征诊断](../results/feature_analysis/expander_decorr_200_exp4_s1407/hard_sample/report.md)
- [Expander all-200 特征多样性分析](../results/feature_analysis/expander_all_200/RRSHID/feature_diversity.md)

## 3. 项目计划完成度矩阵

### 3.1 里程碑完成度

| 里程碑 | 计划内容 | 当前状态 | 证据与说明 |
|---|---|---|---|
| M0 | CUT baseline、VCReg 接入、距离测量协议、基础设施 | **基本完成** | baseline、训练代码、统计指标和特征诊断均已存在；ODCR 差异报告和独立测量协议文档尚未形成 |
| M1 | A1 初版、$\Delta$ 曲线、防御性标定、迁移原型 | **部分完成** | 已有去相关强度扫描、$\Delta$ 静态诊断和有效秩分析；缺少严格 A1、训练曲线、同域参照、检索和迁移实验 |
| M2 | 主表、完整消融、$\Delta$-浓度曲线 | **未完成** | 目前没有完整数据集上的系统主表，也没有浓度分层实验 |
| M3 | 物理引导解耦、退化交换和 DCI/MIG | **未完成** | 尚未发现 `L_guide`、latent swap 或 DCI/MIG 产物 |
| M4 | 线性理论命题与 toy 验证 | **未完成** | 项目文档中有理论草案，但没有对应 toy 实验结果 |
| M5 | 论文内部评审与投稿准备 | **未开始** | 当前更适合整理阶段报告和补实验清单 |

### 3.2 实验项完成度

| 项目文档实验 | 状态 | 判断 |
|---|---|---|
| CUT baseline | **已完成** | 有 checkpoint、完整测试集结果、困难集结果和特征诊断 |
| CUT/FastCUT 参考基线 | **部分完成** | 有 `cut_faster` 可视化结果，但不是完整测试集主表的严格指标 |
| PatchNCE 源域偏好诊断 | **已完成（静态版）** | 已有 $d(A,G(A))$、$d(B,G(A))$ 和 $\Delta$，覆盖多层困难集分析 |
| 去相关快速强度扫描 | **已完成** | 四组 50 epoch 实验完成，Exp4 快速配置胜出 |
| Exp1/Exp4 长程多种子 | **已完成** | 各三种子、200 epoch，已有 PSNR、相关性和有效秩汇总 |
| A1：var-only/cov-only/var+cov | **未完成** | 现有 Exp1/Exp4 都同时启用 covariance 和 variance，只是改变权重强度 |
| A2：权重敏感性 | **部分完成** | 有快速强度扫描和 Exp1/Exp4 两档长程结果，但没有独立且公平的系统网格 |
| A3：施加层位置 | **未完成** | 尚未形成 bottleneck 与中间层的控制变量实验 |
| A4：无引导/两种 `L_guide` | **未完成** | 尚未发现 `L_guide` 实验产物 |
| A5：$\Delta$/域探针/有效秩随 $\lambda_c$ | **部分完成** | 已有静态 $\Delta$ 和有效秩/相关性；域探针与随训练/权重曲线缺失 |
| $\Delta$ 训练曲线 | **未完成** | 当前是训练结束后的离线诊断，不是 epoch 级曲线 |
| $\Delta$-浓度曲线 | **未完成** | 没有按 $\beta$ 或能见度分桶的结果 |
| 同域参照距离 | **未完成** | 没有系统报告 $d(A,A')$、$d(B,B')$ 作为跨域间隔参照 |
| 最近邻检索准确率 | **未完成** | 未发现 paired retrieval 结果 |
| 线性域探针 | **未完成** | 未发现 A/B 分类器准确率或 $G(A)$ 域归属比例 |
| PCA/有效秩谱分析 | **部分完成** | 已有 effective rank、PC1 ratio、相关性；尚未完成完整谱图和 baseline/VCReg 对照协议 |
| 物理引导解耦 | **未完成** | 无 `L_guide`、退化交换或可控生成结果 |
| 跨退化迁移 | **未完成** | 未发现 Rain100L/Rain800/GoPro 训练或下游结果 |
| 检测下游任务 | **未完成** | 未发现 Foggy Cityscapes/RTTS 结果 |
| toy 理论验证 | **未完成** | 未发现 2D/MLP 合成实验产物 |

## 4. 已完成实验结果

### 4.1 CUT baseline 与假对齐诊断

`cut_baseline` 的最佳 checkpoint 为 epoch 165，完整 RRSHID 测试集 PSNR 为 21.0212 dB，困难集 PSNR 为 16.7113 dB。困难集特征诊断显示：

- 清晰域 identity PSNR：23.3107 dB；
- 最深 encoder 层 `enc_resblock_6` 的雾/清晰归一化特征距离：0.9092；
- 最深 PatchNCE 层的 $d(A,G(A))$：0.1728；
- 最深 PatchNCE 层的 $d(B,G(A))$：0.8168；
- 源域偏好 $\Delta_{\text{source}}$：0.6440。

所有已报告的 NCE 层均表现为 $d(A,G(A))<d(B,G(A))$。这支持以下较弱但可靠的表述：

> PatchNCE 的训练对确实把生成结果锚定在雾输入一侧；在配对评估中，生成结果与清晰参考之间存在明显的跨域几何缺口。

这不能单独证明 `G(A)` 没有进入清晰图像分布，因为 GAN 约束的是边际分布，且距离依赖于训练后的特征空间。更准确的名称是“PatchNCE 空间中的源域偏好/假对齐诊断”。

### 4.2 PatchNCE 权重和 identity NCE 频率筛选

30 epoch 低成本筛选结果如下：

| 实验 | 最佳 epoch | 完整测试集 PSNR | 困难集 PSNR |
|---|---:|---:|---:|
| `screen_baseline` | 30 | 19.1107 | 15.1256 |
| `screen_no_idt_nce` | 30 | 18.9271 | 15.0173 |
| `screen_nce_0_5` | 25 | 19.0333 | 15.5697 |
| `screen_nce_2_0` | 15 | 19.7236 | 16.5844 |

该筛选发现高 NCE 权重在短训练预算下改善了困难集，但这一趋势没有在 200 epoch 严格确认中复现。`confirm`（200 epoch、$\lambda_{NCE}=2.0$、identity NCE interval=1）结果为：

- 完整测试集：20.6273 dB；
- 困难集：16.6459 dB；
- 相比 `cut_baseline`：完整集低 0.3939 dB，困难集低约 0.0658 dB。

因此当前不能把高 NCE 作为新的主基线，也不能把困难集筛选结果直接外推到完整训练过程。该实验真正确认的是：**最佳 checkpoint 选择很重要，短程最优不等于长程最优。**

### 4.3 Expander 去相关快速扫描

四组 50 epoch 实验中，`expander_decorr_quick_test_4`（covariance=0.1，variance=0.25）在同预算比较中最好：

- 完整测试集：19.3659 dB；
- 困难集：15.7375 dB；
- 相比快速对照，分别提升 0.1321 dB 和 0.5731 dB。

该结果只用于选择长程配置，不能与 200 epoch baseline 直接比较。它没有完成 var-only/cov-only 分工验证，因为每组都同时启用了两项正则。

### 4.4 Exp1/Exp4 200 epoch 多种子实验

| 实验组 | 协方差权重 | 方差权重 | 最佳 PSNR 均值 ± 标准差 | 最终 PSNR 均值 ± 标准差 | 最终 mean abs correlation | 最终 effective rank |
|---|---:|---:|---:|---:|---:|---:|
| Exp1 | 0.0001 | 0.0025 | 20.8903 ± 0.1181 | 20.8005 ± 0.1965 | 0.1170 ± 0.0184 | 54.7 ± 2.1 |
| Exp4 | 0.1 | 0.25 | 21.1987 ± 0.1251 | 21.1390 ± 0.1048 | 0.1036 ± 0.0011 | 53.5 ± 0.8 |

Exp4 相对 Exp1：

- 最佳 PSNR 平均提升 0.3084 dB；
- 最终 PSNR 平均提升 0.3385 dB；
- mean absolute correlation 从 0.1170 降到 0.1036；
- effective rank 从 54.7 降到 53.5，但没有出现明显的低秩坍缩迹象；
- 三个 seed 的最终 PSNR 均高于对应 Exp1。

Exp4 seed 1407 的最佳 checkpoint 为 epoch 180，完整测试集最佳 PSNR 为 21.3189 dB。困难集特征诊断结果为：

- 翻译 PSNR：17.8587 dB；
- identity PSNR：23.7453 dB；
- `enc_resblock_6` encoder 雾/清晰距离：0.9773；
- $d(A,G(A))=0.1476$；
- $d(B,G(A))=0.7600$；
- $\Delta_{\text{source}}=0.6124$。

相对 `cut_baseline`，Exp4 的源域偏好从 0.6440 降到 0.6124，配对清晰图距离从 0.8168 降到 0.7600，输入雾图距离也从 0.1728 降到 0.1476。这个结果支持“去相关与跨域几何缺口收窄同时发生”，但由于两组长程实验的 `idt_nce_interval` 和部分训练配置并不完全相同，尚不能作为严格的因果消融。

### 4.5 `expander_all_200` 表征多样性分析

`expander_all_200` 已有 304 张 paired 图像的特征多样性报告。`enc_resblock_6` 结果为：

| 分支 | effective rank | mean abs correlation |
|---|---:|---:|
| A fog | 52.19 | 0.1084 |
| B clear | 54.47 | 0.1030 |
| G(A) | 54.27 | 0.1035 |

该结果显示生成分支的表征统计接近清晰分支，且三者的 effective rank 均处于相近水平。这是对表征结构改善的补充证据，但当前提交脚本命名与 checkpoint 目录不完全一致，因此报告中应称为“已有 `expander_all_200` 特征多样性产物”，不应把它直接等同于一个完整的 all-NCE 多种子实验组。

## 5. 对研究假说的当前判断

### H1：退化诱导的相关主导方向

**状态：得到初步支持，但尚未完成机制证明。**

支持证据：

- 去相关正则降低了 mean absolute correlation；
- Exp4 的有效秩没有崩塌；
- 去相关后 $d(B,G(A))$ 和 $\Delta$ 在困难集上下降；
- `expander_all_200` 中 A、B、G(A) 的深层统计更接近。

尚缺证据：

- 协方差谱或 PCA 主方向的直接对照；
- $\Delta$ 随雾浓度 $\beta$ 变化的曲线；
- cov-only 与 var-only 的分工实验；
- 排除“整体 embedding 收缩”这一解释的归一化分析。

因此论文中应使用“consistent with”或“支持该机制假说的证据”，不要写成“证明 covariance 项专门压缩了雾方向”。

### H2：variance/covariance/PatchNCE 三者分工

**状态：尚未验证。**

已有实验只比较了两档同时开启的 covariance+variance 权重。没有独立的 covariance-only、variance-only 和严格匹配的 full 组合，因此目前只能说“Exp4 的联合去相关设置有效”，不能说明三项损失分别承担了什么作用。

### 假对齐与 $\Delta$ 指标

**状态：静态诊断已完成，动态和防御性验证未完成。**

当前数据足以支持把 $\Delta$ 作为一个有用诊断指标，但需要补充：

- 同域跨场景距离 $d(A,A')$、$d(B,B')$；
- paired retrieval top-1/top-k；
- 每个 epoch 的 $\Delta$ 曲线；
- 完整测试集而非仅困难集的统计置信区间。

## 6. 当前不能宣称的内容

基于现有证据，以下表述目前不成立或证据不足：

1. 已经完成了 content/degradation 的显式解耦；
2. VCReg 已经在完整公平消融中证明优于 CUT；
3. covariance 项单独负责消除退化主方向；
4. 已经验证 $\Delta$ 随雾浓度增加而增加；
5. 已经证明域探针准确率下降或 $G(A)$ 转入 B 域；
6. 已经证明表征可跨去雨、去模糊迁移；
7. 已经完成物理引导的 latent swap、DCI/MIG 或理论 toy 验证；
8. `expander_all_200` 已经是配置和命名完全可追溯的 all-NCE 多种子主实验。

## 7. 建议的下一阶段最小实验集

### 优先级 1：补齐严格 A1

固定同一训练预算、seed、`lambda_nce`、identity NCE interval、数据增强和 checkpoint 选择规则，只改变去相关项：

1. no VCReg；
2. covariance-only；
3. variance-only；
4. covariance+variance（Exp4）。

每组至少 3 个 seed，报告完整测试集和困难集 PSNR、mean absolute correlation、off-diagonal energy、effective rank、PC1 ratio，以及 $d(A,G(A))$、$d(B,G(A))$、$\Delta$。

### 优先级 2：补齐 $\Delta$ 的防御性标定

在相同的 `analyze_features.py` 协议下增加：

- $d(A,A')$ 与 $d(B,B')$ 的同域参照；
- paired retrieval top-1/top-5；
- 完整 304 张测试图统计，而不是只报告 47 张困难集；
- bootstrap 置信区间或至少逐图均值/标准差。

### 优先级 3：补齐机制指标

- 保存 epoch 级 checkpoint 或在验证阶段定期计算 $\Delta$；
- 按合成雾参数 $\beta$/能见度分桶；
- 训练冻结 encoder 的线性 A/B 域探针；
- 输出 A、B、G(A) 的完整协方差谱和有效秩曲线。

### 优先级 4：再决定是否做解耦和迁移

只有在严格 A1 和机制指标稳定后，才投入 `L_guide`、latent swap、DCI/MIG、Rain/GoPro 迁移实验。否则容易把“通道去相关”过早包装成“功能性解耦”。

## 8. 阶段性论文叙事建议

当前最稳妥的 Paper 1 叙事是：

> 在非配对 CUT 去雾中，PatchNCE 使生成结果在训练特征空间中保持明显的源域偏好，形成可测量的跨域逐样本对齐缺口。将 VCReg 风格的 Expander 去相关正则加入 fine-tuning 后，可以降低通道相关性，并在多种子长程实验中改善去雾 PSNR，同时缩小该缺口。现阶段把它定位为“跨域表征几何诊断与去相关正则化”，而不是已经完成的通用 disentanglement 或 degradation-invariant pretraining。

这条叙事与当前证据最匹配，也为后续严格消融和迁移实验保留空间。

## 9. 最终完成度判断

| 方向 | 完成度 | 结论 |
|---|---:|---|
| CUT 去雾 baseline | 100% | 可作为稳定对照 |
| PatchNCE 源域偏好诊断 | 80% | 静态困难集诊断完成，完整防御性协议未完成 |
| Expander VCReg 接入 | 100% | 已跑通并保留多种子长程结果 |
| 去相关强度筛选 | 100% | 快速筛选与 Exp1/Exp4 长程实验完成 |
| 严格 A1 分工消融 | 0% | 尚未开始 |
| $\Delta$ 动态/浓度机制验证 | 20% | 只有静态结果 |
| 显式解耦与物理引导 | 0% | 尚未开始 |
| 跨退化迁移 | 0% | 尚未开始 |
| 理论 toy 验证 | 0% | 尚未开始 |

**阶段结论：** 当前项目已经完成了一个有价值的诊断与正则化原型，足以进入“严格公平消融 + 机制验证”阶段；还不能宣称项目文档中的完整解耦方案和跨退化表征学习目标已经完成。
