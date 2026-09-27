# Encoder 原始空间退化稳定性完整实验计划

## 1. 计划定位

本计划将[Encoder 原始空间退化稳定性实验需求](Encoder原始空间退化稳定性实验需求.md)落实为可执行实验流程。

本阶段只进行离线推理和表示分析，不重新训练模型，不修改训练损失，不使用 B 域图像参与生成退化或模型选择。

核心问题：

> 相比标准 CUT，VCReg 是否让 Encoder 原始 feature 对可控的非均匀雾退化和轻微常见图像扰动更加稳定，同时不牺牲跨图像内容可分性？

主模型：

| 代号 | 实验名 | 用途 |
| --- | --- | --- |
| CUT | `cut_baseline` | 标准 CUT 对照 |
| VCReg | `expander_decorr_200_exp1_s1407` | 当前 VCReg 对照 |

两组模型必须分别加载已有最佳 checkpoint，但共享同一批原图、随机场、退化参数和预处理结果。

## 2. 总体流程

```text
P0 环境与 checkpoint 核验
  -> P1 退化模块数值与视觉校准
  -> P2 同一扰动集生成与数据清单冻结
  -> P3 Encoder 原始 feature smoke test
  -> P4 CUT/VCReg 全量离线推理
  -> P5 汇总统计、曲线和防坍缩检查
  -> P6 结果复核与阶段结论
```

任何阶段失败时，只修复该阶段，不使用未通过验收的结果推进后续比较。

## 3. P0：环境与 checkpoint 核验

### 3.1 环境

使用项目当前 Python 环境：

```bash
/home/visitor1/.conda/envs/dehaze/bin/python
```

运行目录固定为仓库根目录 `/home/visitor1/Decorrelation`。

### 3.2 数据核验

核验 `datasets/RRSHID/testA` 存在且包含有效 PNG；全量图像数量记录为实际有效数量，预期为 304；文件名排序固定；每张图均可转换为 RGB；图像尺寸和 padding 规则能够被记录。

### 3.3 模型核验

确认以下 checkpoint 存在且能加载：

```text
results/ckpt/cut_baseline/best_test_psnr.pt
results/ckpt/expander_decorr_200_exp1_s1407/best_test_psnr.pt
```

加载后记录 checkpoint epoch、`ngf`、`encoder_blocks`、downsample mode、state dict 完整性和实际可返回的 Encoder 层名。

指定层固定为 `stem`、`downsample_1`、`downsample_2`、第一层 ResBlock 和最后一层 ResBlock。最后一层名称必须根据实际 `encoder_blocks` 动态确定。

### 3.4 P0 验收

- 两个 checkpoint 均可加载；
- 两个模型均能对同一张 RGB 图像完成 `encode(..., return_features=True)`；
- 指定层均存在；
- 输入归一化和 padding 在两个模型间一致。

## 4. P1：退化协议校准

### 4.1 ASM 配置冻结

退化实现使用 [utils/degradation.py](../utils/degradation.py)。正式实验必须把实际参数写入 manifest，不依赖代码默认值。

建议第一版固定：

```text
field_size = 8
d_range = [0.3, 1.0]
beta_range = [0.05, 0.6]
coupling = 0.0
atmospheric_light = [0.90, 0.93, 0.96]
interpolation = bicubic
```

ASM 强度只改变 `beta_scale`，不改变 `d_field`、`beta_field`：

```text
clean       0.00
weak        0.25
medium      0.50
strong      0.75
very_strong 1.00
```

### 4.2 随机实例设计

主实验使用固定的每图像 seed：`field_seed = 1407 + image_index`。强度扫描固定 field seed。随机实例方差实验使用至少 5 组 base seed：`1407, 2407, 3407, 4407, 5407`。

主实验使用 `coupling=0.0`。只有当主结果显示随机场方差过大时，才增加 coupling 消融；不得在主实验中同时改变 coupling、field size 和 beta 范围。

### 4.3 常见扰动配置

| 条件 | 参数 |
| --- | --- |
| `blur_02` | Gaussian sigma 0.2 |
| `blur_05` | Gaussian sigma 0.5 |
| `blur_08` | Gaussian sigma 0.8 |
| `gamma_dark` | gamma 1.1 |
| `gamma_bright` | gamma 0.9 |
| `noise` | Gaussian std 0.025，seed 1407 |
| `contrast_low` | contrast 0.9 |
| `contrast_high` | contrast 1.1 |

组合测试至少包含 `ASM medium`、`blur_05`、`gamma_bright`、`ASM medium + blur_05`、`ASM medium + gamma_bright` 和 `ASM medium + blur_05 + gamma_bright`。组合顺序固定为 `ASM -> blur -> gamma -> noise -> contrast`。

### 4.4 P1 验收

- `clean` 输出与输入逐元素相等；
- 相同 seed 生成的场和图像逐元素相等；
- transmission 位于 `(0,1]`；
- 常见扰动输出位于 `[-1,1]`；
- medium ASM 视觉上可见但不整体过曝；
- blur 边缘无零填充暗边；
- 通过后冻结参数，不在 CUT/VCReg 之间重新调参。

## 5. P2：扰动样本和 manifest 冻结

建议输出目录：

```text
results/feature_analysis/encoder_robustness/shared_inputs/
```

保存 `manifest.json`、每张图像每个 condition 的 PNG，以及 ASM 的 depth、beta、transmission 场。manifest 至少记录 git commit 或工作树标识、数据集路径和图像列表、每图像 field seed、ASM 与常见扰动完整配置、组合顺序、输入尺寸、RGB 转换、归一化、padding、生成脚本版本和生成时间。

CUT 与 VCReg 的推理只能读取这份 shared manifest，禁止各自生成一份扰动图。每个样本必须由 `image_name + condition + seed + parameter_record` 重建。

## 6. P3：Encoder 原始 feature smoke test

使用 3 张图像、`clean/weak/medium` 三种 ASM 条件和所有指定层完成小规模运行。

检查原图和退化图 feature shape、patch 数量一致；feature 无 NaN/Inf；cosine 和 normalized L2 有限；clean 到 clean 距离接近 0；两个模型没有误用对方 feature。

人工小例子必须验证：相同向量 cosine distance 为 0，正交向量约为 1，反向向量约为 2，相同 feature 的 normalized L2 为 0。

## 7. P4：正式离线推理矩阵

### 7.1 ASM 强度矩阵

对两个模型、304 张图像、5 个 ASM 强度、5 个 Encoder 层运行，共 `2 x 304 x 5 x 5` 个模型-图像-条件-层单元。每个单元输出 cosine、normalized L2、norm ratio 和图像空间 normalized pixel L2。

### 7.2 随机实例矩阵

固定 `medium`，使用 5 个 base seed，比较两个模型、304 张图像、5 个实例、5 个层，共 `2 x 304 x 5 x 5` 个单元，用于估计 seed 间方差，不用于挑选对 VCReg 有利的 seed。

### 7.3 常见扰动矩阵

对两个模型、304 张图像、单项和组合条件、5 个层运行。ASM 组合必须复用 P2 的 medium 样本。

### 7.4 推荐分批

按 3 张 smoke test、304 张 ASM 强度、304 张随机实例、304 张常见扰动单项、304 张组合扰动、防坍缩统计的顺序分批。每批完成后立即保存 JSONL 中间结果。

## 8. P5：统计与比较

每条 `per_image_metrics.jsonl` 使用唯一键 `model, image_name, condition, field_seed, layer`，并记录 cosine distance、normalized L2、feature norm ratio、image pixel L2、feature shape 和输入输出校验信息。

每个模型、条件和层报告 mean、std、P05、P50、P95、bootstrap 95% CI、VCReg-CUT 绝对差值、相对变化率和逐图配对胜率。

配对主检验为：

$$
\Delta_l(c)=d_{l,\mathrm{VCReg}}(c)-d_{l,\mathrm{CUT}}(c)
$$

报告 paired mean difference、paired bootstrap 95% CI、Wilcoxon signed-rank p 值和效应量。多层、多条件检验使用 Benjamini-Hochberg FDR；探索性结果标注为探索性。

对固定 field seed 绘制 `beta_scale -> distance` 曲线，报告均值、bootstrap CI、分段斜率、单调性违反比例和 CUT/VCReg 曲线面积差。

## 9. P5 防坍缩与内容保持检查

对 clean 和 representative medium feature 计算每通道标准差、covariance effective rank、top-1/top-5 PCA variance ratio、不同原图之间的同层 cosine 距离，以及同图 clean/perturbed 距离与跨图 clean/clean 距离之比。

扰动距离下降而跨图内容距离保持，才支持稳定性改善。扰动距离和跨图内容距离同时下降，优先怀疑 collapse。effective rank 明显下降但扰动距离变小，不得宣称鲁棒性改善。norm ratio 偏离 1 时，不能只依据 cosine 下结论。

## 10. P6 阶段结论规则

支持 VCReg 的最低证据：weak/medium ASM 条件下多数指定层 paired cosine difference 小于 0；normalized L2 没有明显反向恶化；多个 field seed 方向一致；bootstrap CI 或效应量支持差异不是单图异常；clean 跨图距离和 effective rank 没有明显坍缩；常见扰动上至少有部分一致改善。

若只有 ASM 改善，只能声称雾特异性稳定性；若只有常见扰动改善，不能声称获得雾退化不变性；若结果高度依赖 seed，应增加随机场实例；若差异很小，应报告阴性结果并转向域 probe 或 paired retrieval。

## 11. 最终产物

```text
results/feature_analysis/encoder_robustness/
  shared_inputs/manifest.json
  <experiment>/manifest.json
  <experiment>/per_image_metrics.jsonl
  <experiment>/summary.json
  <experiment>/curves.csv
  <experiment>/plots/
  <experiment>/summary.md
```

文档产物为本计划和 [Encoder退化稳定性分析报告](Encoder退化稳定性分析报告.md)。参数变化必须追加变更记录，不覆盖旧 manifest。

## 12. 禁止事项

- 不为两个模型分别重新随机生成扰动；
- 不使用 B 域图像参与扰动生成；
- 不用 Expander NCE 距离替代 raw Encoder 指标；
- 不用最终 epoch 替代验证集最佳 checkpoint；
- 不因某个 seed 或某一层结果好看而挑选报告范围；
- 不把相关性下降、距离下降直接写成已完成的跨域不变性；必须结合域探针、同域参照和跨退化迁移进行验证。
