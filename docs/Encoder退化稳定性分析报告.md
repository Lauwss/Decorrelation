# Encoder 原始空间退化稳定性分析报告

> 报告状态：第一轮 304 张图像实验已完成；随机场多 seed、防坍缩统计和扩展 blur 强度仍待完成。
>
> 本报告只记录按[完整实验计划](Encoder退化稳定性完整实验计划.md)实际运行得到的结果，不把预期结果写成实验结论。

## 1. 执行摘要

### 1.1 当前结论

第一轮正式运行覆盖 RRSHID `testA` 的 304 张图像、两个 checkpoint、5 个 raw Encoder 层和 10 个退化条件。结果显示：VCReg 在当前 ASM 随机雾协议下的 raw Encoder 距离高于 CUT，但在 `blur_05`、`noise` 和 `contrast_low` 上距离更低；因此当前证据不支持“VCReg 对所有退化都更稳定”的结论。

### 1.2 研究问题

比较 CUT（`cut_baseline`）和 VCReg（`expander_decorr_200_exp1_s1407`）。主指标为 Encoder 原始 feature 的同位置 patch cosine distance 和 normalized L2 distance。PatchNCE Expander 距离不作为本报告主指标。

## 2. 实验版本与证据边界

| 项目 | 实际值 |
| --- | --- |
| 报告日期 | 待填写 |
| 仓库 commit | 待填写 |
| Python 环境 | `/home/visitor1/.conda/envs/dehaze/bin/python` |
| 数据集 | `datasets/RRSHID/testA` |
| 图像数量 | 待填写，预期 304 |
| 退化模块 | `utils/degradation.py` |
| shared manifest | 待填写 |
| 统计脚本 | `analyze_encoder_robustness.py` |
| 第一轮输出 | `results/feature_analysis/encoder_robustness/full/` |

正式结论必须能追溯到 checkpoint、shared manifest、单图 JSONL 和汇总脚本。无法追溯参数的图表只能作为探索性可视化。

## 3. 模型与数据核验

### 3.1 Checkpoint

| 模型 | checkpoint | epoch | 配置 | 加载状态 |
| --- | --- | ---: | --- | --- |
| CUT | `results/ckpt/cut_baseline/best_test_psnr.pt` | 待填写 | 待填写 | 待填写 |
| VCReg | `results/ckpt/expander_decorr_200_exp1_s1407/best_test_psnr.pt` | 待填写 | 待填写 | 待填写 |

实际加载 epoch：CUT 为 165，VCReg 为 195。两个模型使用 `ngf=64`、`encoder_blocks=6`、`downsample_mode=antialias`。

### 3.2 Encoder 层

| 层 | feature shape | patch 数量 | 备注 |
| --- | --- | ---: | --- |
| `stem` | 待填写 | 待填写 | |
| `downsample_1` | 待填写 | 待填写 | |
| `downsample_2` | 待填写 | 待填写 | |
| `enc_resblock_1` | 待填写 | 待填写 | |
| `enc_resblock_6` | 待填写 | 待填写 | |

## 4. 退化协议记录

### 4.1 ASM 参数

| 参数 | 实际值 |
| --- | --- |
| field type | 待填写 |
| field size | 待填写 |
| d range | 待填写 |
| beta range | 待填写 |
| beta scales | `0.00, 0.25, 0.50, 0.75, 1.00` |
| coupling | 待填写 |
| atmospheric light | 待填写 |
| interpolation | 待填写 |
| base seeds | 待填写 |

ASM 定义为：

$$
t(x)=\exp(-\beta(x)d(x)),
\qquad
I(x)=J(x)t(x)+A(1-t(x)).
$$

### 4.2 常见扰动参数

| 条件 | 参数 | 实际状态 |
| --- | --- | --- |
| `blur_02` | sigma 0.2 | 待填写 |
| `blur_05` | sigma 0.5 | 待填写 |
| `blur_08` | sigma 0.8 | 待填写 |
| `gamma_dark` | gamma 1.1 | 待填写 |
| `gamma_bright` | gamma 0.9 | 待填写 |
| `noise` | std 0.025 | 待填写 |
| `contrast_low` | contrast 0.9 | 待填写 |
| `contrast_high` | contrast 1.1 | 待填写 |
| `ASM + blur + gamma` | medium + sigma 0.5 + gamma 0.9 | 待填写 |

## 5. 输入质量与退化校准

### 5.1 图像级校准

| 条件 | transmission P05/P50/P95 | mean pixel change | 越界数量 | 结论 |
| --- | --- | ---: | ---: | --- |
| clean | 待填写 | 0 | 待填写 | 待填写 |
| ASM weak | 待填写 | 待填写 | 待填写 | 待填写 |
| ASM medium | 待填写 | 待填写 | 待填写 | 待填写 |
| ASM strong | 待填写 | 待填写 | 待填写 | 待填写 |
| ASM very strong | 待填写 | 待填写 | 待填写 | 待填写 |

### 5.2 可复现性

- 相同 seed 是否逐元素一致：待填写；
- CUT/VCReg 是否读取同一 shared manifest：待填写；
- clean 是否严格恒等：待填写；
- 是否存在缺失或重复图像：待填写。

## 6. 方法与指标

对层 $l$ 和位置 $i$，令原图 feature 为 $f_{l,i}$，扰动图 feature 为 $f'_{l,i}$。

### 6.1 Cosine distance

$$
d^{\mathrm{cos}}_l(x,x')=
\frac{1}{N_l}\sum_i
\left(1-\hat f_{l,i}^{\mathsf T}\hat f'_{l,i}\right).
$$

### 6.2 Normalized L2

$$
d^{\mathrm{normL2}}_l(x,x')=
\frac{1}{N_l}\sum_i
\frac{\lVert f_{l,i}-f'_{l,i}\rVert_2}
{(\lVert f_{l,i}\rVert_2+\lVert f'_{l,i}\rVert_2)/2+\epsilon}.
$$

### 6.3 配对模型差异

$$
\Delta_l(c)=d_{l,\mathrm{VCReg}}(c)-d_{l,\mathrm{CUT}}(c).
$$

$\Delta_l(c)<0$ 表示 VCReg 在该条件和层上对退化更稳定，但只有在防坍缩指标没有恶化时才具有正向解释。

## 7. ASM 强度响应结果

### 7.1 Cosine distance

| 层 | 模型 | clean | weak | medium | strong | very strong |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `stem` | CUT | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `stem` | VCReg | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `downsample_1` | CUT | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `downsample_1` | VCReg | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `downsample_2` | CUT | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `downsample_2` | VCReg | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `enc_resblock_1` | CUT | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `enc_resblock_1` | VCReg | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `enc_resblock_6` | CUT | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `enc_resblock_6` | VCReg | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

### 7.2 VCReg-CUT 差值

| 层 | weak | medium | strong | 结论 |
| --- | ---: | ---: | ---: | --- |
| `stem` | 待填写 | 待填写 | 待填写 | 待填写 |
| `downsample_1` | 待填写 | 待填写 | 待填写 | 待填写 |
| `downsample_2` | 待填写 | 待填写 | 待填写 | 待填写 |
| `enc_resblock_1` | 待填写 | 待填写 | 待填写 | 待填写 |
| `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 |

### 7.3 响应曲线

待插入：`plots/asm_cosine_by_layer.png`、`plots/asm_normalized_l2_by_layer.png`、`plots/asm_model_difference.png`。

观察记录：

- 是否随 beta_scale 单调增加：待填写；
- 是否存在低强度跳变：待填写；
- VCReg 曲线是否更平滑：待填写；
- 随机场 seed 间方差：待填写。

### 7.4 第一轮实际结果

第一轮固定 `field_seed = 1407 + sorted image index`，每张图只使用一个随机场实例。raw Encoder cosine distance 的全量均值如下：

| 条件 | 层 | CUT | VCReg | VCReg-CUT | VCReg 更低比例 |
| --- | --- | ---: | ---: | ---: | ---: |
| ASM weak | `stem` | 0.011843 | 0.016029 | +0.004186 | 0.7% |
| ASM weak | `enc_resblock_6` | 0.015741 | 0.023924 | +0.008183 | 0.3% |
| ASM medium | `stem` | 0.031834 | 0.042481 | +0.010647 | 0.0% |
| ASM medium | `enc_resblock_6` | 0.041580 | 0.060196 | +0.018616 | 0.7% |
| ASM strong | `stem` | 0.053048 | 0.069696 | +0.016648 | 0.0% |
| ASM strong | `enc_resblock_6` | 0.068782 | 0.096644 | +0.027862 | 0.7% |
| ASM very strong | `stem` | 0.073683 | 0.095437 | +0.021754 | 0.3% |
| ASM very strong | `enc_resblock_6` | 0.095228 | 0.130909 | +0.035681 | 1.0% |
| blur 0.5 | `enc_resblock_6` | 0.003783 | 0.003087 | -0.000696 | 84.2% |
| noise | `enc_resblock_6` | 0.195281 | 0.132288 | -0.062993 | 100.0% |
| contrast 0.9 | `enc_resblock_6` | 0.000048 | 0.000024 | -0.000024 | 100.0% |
| ASM medium + blur + gamma | `enc_resblock_6` | 0.048424 | 0.066375 | +0.017951 | 1.6% |

全量逐图结果和五层完整表位于 [full/summary.md](../results/feature_analysis/encoder_robustness/full/summary.md)，逐图数据位于 `full/per_image_metrics.jsonl`。

当前最重要的观察是：ASM 距离随强度增加，且 VCReg 在五个层上均高于 CUT；相反，VCReg 对噪声和轻微对比度扰动更稳定。ASM 与 blur/gamma 组合仍主要由 ASM 方向决定，VCReg 距离高于 CUT。

## 8. 常见扰动结果

| 条件 | 层 | CUT cosine | VCReg cosine | 差值 | CUT normL2 | VCReg normL2 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| blur 0.2 | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| blur 0.5 | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| blur 0.8 | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| gamma 0.9 | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| gamma 1.1 | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| noise | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| contrast 0.9 | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| contrast 1.1 | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| ASM + blur + gamma | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

完整层级表和图表路径待正式运行后补充。

## 9. 随机实例稳定性

固定 `beta_scale=0.5`，比较 base seed `1407/2407/3407/4407/5407`。

| 层 | 模型 | seed mean | seed std | P05 | P50 | P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `stem` | CUT | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `stem` | VCReg | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `downsample_2` | CUT | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `downsample_2` | VCReg | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `enc_resblock_6` | CUT | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| `enc_resblock_6` | VCReg | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

## 10. 防坍缩与内容保持

| 指标 | 层 | CUT | VCReg | 相对变化 | 解释 |
| --- | --- | ---: | ---: | ---: | --- |
| channel std | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 |
| effective rank | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 |
| PCA top-1 ratio | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 |
| clean 跨图 cosine | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 |
| medium 跨图 cosine | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 |

必须回答：

1. VCReg 的扰动距离下降时，clean 跨图内容距离是否保持？
2. effective rank 是否明显下降？
3. feature norm ratio 是否偏离 1？
4. 是否存在“所有图像都更相似”的替代解释？

## 11. 统计显著性与效应量

| 条件 | 层 | paired mean difference | 95% CI | Wilcoxon p | FDR q | VCReg 胜率 |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| ASM weak | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| ASM medium | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| ASM strong | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |
| ASM + blur + gamma | `enc_resblock_6` | 待填写 | 待填写 | 待填写 | 待填写 | 待填写 |

不能只根据 p 值做判断。样本量大时，必须同时报告差值大小、置信区间和胜率。

## 12. 阶段结论

### 12.1 可以支持的结论

在 RRSHID `testA`、当前固定随机场协议和第一轮条件下，可以支持以下有限结论：

1. 两个模型均能输出有限且可复现的 raw Encoder 距离；clean 条件严格为零距离。
2. ASM 强度增加时，两个模型的 raw Encoder cosine distance 都整体增加，说明该退化确实被 Encoder 感知。
3. 当前 VCReg checkpoint 在 ASM weak/medium/strong/very strong 上均比 CUT 产生更大的同位置 cosine distance，不能把 VCReg 表述为当前 ASM 协议下的更强雾不变表示。
4. 当前 VCReg 在 `blur_05`、`noise` 和 `contrast_low` 上的 cosine distance 低于 CUT，说明其改善可能具有退化类型依赖性。

这些结论仅适用于第一轮单 seed、当前参数和 raw Encoder cosine/normL2 指标。

### 12.2 不能支持的结论

在没有额外实验前，不得从本报告推出：已完成完整的跨域不变表征学习；已证明 VCReg 对真实物理深度变化具有不变性；已证明模型能泛化到雨、模糊或其他退化；已证明 cosine distance 下降不是 feature collapse；已证明 Expander 空间变化等同于 raw Encoder 空间变化。

### 12.3 后续实验建议

| 结果模式 | 后续实验 |
| --- | --- |
| ASM 和常见扰动都改善 | 做 paired retrieval 和线性域 probe |
| 只有 ASM 改善 | 增加随机场实例和 beta 强度，检查雾特异性 |
| 只有浅层改善 | 分析归一化、下采样和局部纹理响应 |
| 只有深层改善 | 检查内容保持与跨域几何缺口 |
| 距离降低伴随 rank 降低 | 做 feature collapse 防御实验，不下鲁棒性结论 |
| 无明显差异 | 报告阴性结果，转向训练动态或 decoder 机制 |

## 13. 变更记录

| 日期 | 变更 | 原因 | 是否重新运行 |
| --- | --- | --- | --- |
| 2026-09-26 | 创建报告模板 | 建立第一阶段结果承载文档 | 否 |
| 2026-09-26 | 完成第一轮 304 张 raw Encoder 实验 | 获得 CUT/VCReg 在 10 个条件下的逐图和汇总结果 | 是 |
