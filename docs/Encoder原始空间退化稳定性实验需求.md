# 第一阶段：Encoder 原始空间退化稳定性实验需求

## 1. 实验目标

本阶段只做离线表示分析，不修改训练目标，不重新训练模型，不使用 B 域图像参与训练。

核心问题是：

> VCReg 是否让 Encoder 对语义保持的雾退化和轻微图像扰动更加稳定，从而减少模型对局部纹理、对比度或雾纹理 shortcut 的依赖？

最终关注的是 Encoder 原始 feature 空间，而不是 Expander 投影空间：

$$
F_l(x)=E_l(x)
$$

主比较对象为：

- 标准 CUT：`cut_baseline`
- VCReg：`expander_decorr_200_exp1_s1407`

两组模型必须使用各自训练完成的 checkpoint，但使用完全相同的输入图像、退化参数、图像预处理和统计协议。

## 2. 实验边界

### 2.1 本阶段包含

- 对 A 域图像生成可控的 ASM 非均匀雾退化版本；
- 对随机退化场的强度和随机实例分别做扫描；
- 叠加轻微模糊、gamma、噪声或对比度扰动的独立测试；
- 提取 Encoder 原始 feature；
- 计算同一空间位置的 cosine distance 和 normalized L2 distance；
- 比较 CUT 与 VCReg 的层级响应曲线、扰动敏感度和稳定性。

### 2.2 本阶段不包含

- 不使用 Expander 距离作为主结论；
- 不把固定 Expander 作为模型优劣证明；
- 不加入 A/B 配对损失或任何新的训练损失；
- 不训练域分类器、检索器或新的 Encoder；
- 不把距离变小直接解释为语义变强，必须同时检查内容保持和 feature collapse 风险。

## 3. 数据与模型

### 3.1 数据

默认使用：

```text
datasets/RRSHID/testA
```

只使用 A 域图像生成扰动版本。默认应覆盖全部有效 PNG；调试阶段允许使用 `--max-images` 限制样本数。

原图及其所有扰动版本必须共享：

- 相同 resize 和 padding 规则；
- 相同空间尺寸；
- 相同裁剪窗口；
- 相同随机种子；
- 相同的随机场种子和随机场生成参数；
- 相同的 ASM 大气光参数；
- 相同的退化强度参数。

### 3.2 模型

两组模型均加载：

```text
results/ckpt/<experiment>/best_test_psnr.pt
```

Encoder 配置必须与 checkpoint 匹配。默认读取：

```text
ngf = 64
encoder_blocks = 6
downsample_mode = antialias
```

分析层固定为：

```text
stem
downsample_1
downsample_2
enc_resblock_1
enc_resblock_6
```

如果 checkpoint 使用不同的 `encoder_blocks`，最后一层名称必须随配置变化，不能硬编码为不存在的层。

## 4. 退化生成协议

### 4.1 ASM 基础模型

使用大气散射模型生成额外退化：

$$
I(x)=J(x)t(x)+A(1-t(x))
$$

其中 $J$ 是输入 A 图像，$A$ 是大气光，$t(x)$ 是 transmission map。

本实验不估计真实深度，也不使用单目深度估计结果。$d(x)$ 被定义为由固定随机种子生成的连续随机深度场，表示退化生成器中的相对光程差，而不是物理深度测量。$\beta(x)$ 同样是由随机场生成器产生的非均匀雾密度场。基础 transmission 为：

$$
t(x)=\exp(-\beta(x)d(x))
$$

其中 $d(x)\geq 0$、$\beta(x)\geq 0$，并且二者均为低频、连续、空间非均匀的随机场。退化生成器必须提供明确的独立接口，返回 $d(x)$、$\beta(x)$、$t(x)$ 和最终图像 $I(x)$。同一图像在 CUT 与 VCReg 评估中必须复用完全相同的随机场，不能分别重新采样。

### 4.2 随机场生成

不预设左、右、上、下等空间方向，也不要求雾集中在某个手工指定区域。第一版使用随机场自然产生非均匀雾：

| 随机场 | 生成要求 |
| --- | --- |
| `d_field` | 从低分辨率随机噪声采样，经双线性上采样和可选高斯平滑，归一化到指定范围 |
| `beta_field` | 独立生成低频随机噪声场，经正值化、归一化后作为非均匀雾密度形状 |
| `coupled_field` | 可选：使用一部分共享低频噪声，使 $d(x)$ 与 $\beta(x)$ 存在可控相关性 |

不能直接把逐像素白噪声作为 $d(x)$ 或 $\beta(x)$。随机场的最低要求是低频连续，并通过配置记录基础分辨率、插值方式、平滑参数、分布类型、均值、标准差和随机种子。每个图像使用独立但可复现的 `field_seed`。

建议将随机场先标准化为 $z(x)\in[0,1]$，再映射到配置范围：

$$
d(x)=d_{\min}+(d_{\max}-d_{\min})z_d(x)
$$

$$
\beta(x)=\beta_{\mathrm{scale}}\left[\beta_{\min}+
(\beta_{\max}-\beta_{\min})z_\beta(x)\right]
$$

其中 $z_d(x)$ 和 $z_\beta(x)$ 可以独立采样，也可以通过 `coupling` 参数控制相关程度。默认使用独立随机场，避免把某一种固定相关结构误认为 VCReg 的优势。

### 4.3 强度等级

第一版至少使用五个强度等级：

```text
clean       beta_scale = 0.00
weak        beta_scale = 0.25
medium      beta_scale = 0.50
strong      beta_scale = 0.75
very_strong beta_scale = 1.00
```

`beta_scale` 是退化强度的主控制量，直接缩放 $\beta(x)$；$d(x)$ 的随机场范围在一次实验中保持不变。具体的 $d$、基础 $\beta$ 范围和大气光 $A$ 必须在 manifest 中记录。强度实验改变 `beta_scale`，不重新生成随机场，这样才能得到同一退化实例随强度变化的响应曲线。

`clean` 必须保留为同一张原图的零扰动参考，而不是重新保存后再次压缩的图像。

为区分“强度效应”和“随机实例效应”，至少使用两种协议：

1. 固定 `field_seed`，扫描 `beta_scale`；
2. 固定 `beta_scale`，使用多个 `field_seed`，报告随机实例间的均值和方差。

### 4.4 轻微常见扰动

雾退化和常见扰动必须分开评估，再评估组合效果。

| 扰动 | 第一版参数 |
| --- | --- |
| `blur` | Gaussian blur $\sigma\in\{0.2,0.5,0.8\}$ |
| `gamma` | $\gamma\in\{0.9,1.0,1.1\}$ |
| `noise` | 小幅 Gaussian noise，记录标准差 |
| `contrast` | 小幅 contrast scale，记录缩放值 |

组合测试至少包含：

```text
ASM only
blur only
gamma only
ASM + blur
ASM + gamma
ASM + blur + gamma
```

组合扰动必须使用小参数，确保主体、边缘和场景布局仍然可辨认。每种组合的参数应固定，不允许对每张图独立随机改变而不记录。

## 5. Encoder 特征与距离计算

### 5.1 原始 feature

对原图和扰动图分别调用：

```python
generator.encode(image, return_features=True)
```

不经过 `PatchSampleF`、Expander 或训练时的 NCE projection。

对层 $l$ 的 feature map：

$$
F_l\in\mathbb{R}^{B\times C_l\times H_l\times W_l}
$$

将其转换为：

$$
F_l\rightarrow[B,H_lW_l,C_l]
$$

每个空间位置对应一个原始 Encoder feature 向量。

### 5.2 Patch cosine distance

对每个空间位置单独 L2 归一化：

$$
\hat f_{l,i}=\frac{f_{l,i}}{\lVert f_{l,i}\rVert_2+\epsilon}
$$

原图 $x$ 与扰动图 $x'$ 的单图、单层距离为：

$$
d^{\mathrm{cos}}_l(x,x')=
\frac{1}{N_l}\sum_{i=1}^{N_l}
\left(1-\hat f_{l,i}(x)^\mathsf{T}\hat f_{l,i}(x')\right)
$$

比较必须使用相同空间位置，不能使用最近邻 patch 匹配。

### 5.3 Normalized L2 distance

为补充 cosine 对 feature norm 不敏感的问题，同时计算：

$$
d^{\mathrm{normL2}}_l(x,x')=
\frac{1}{N_l}\sum_{i=1}^{N_l}
\frac{\lVert f_{l,i}(x)-f_{l,i}(x')\rVert_2}
{\frac{1}{2}\left(\lVert f_{l,i}(x)\rVert_2+\lVert f_{l,i}(x')\rVert_2\right)+\epsilon}
$$

同时记录 feature norm ratio：

$$
r_l(x,x')=\frac{\lVert f_l(x')\rVert_2}{\lVert f_l(x)\rVert_2+\epsilon}
$$

### 5.4 原图到扰动图的响应曲线

对每个图像、随机场实例和层，记录：

$$
r_l(\alpha)=d_l(x,x_\alpha)
$$

并对全部图像求均值、标准差、P05、P50 和 P95。

需要检查：

- 小强度是否出现异常跳变；
- 距离是否随强度大致平滑增长；
- 不同随机场实例的响应是否稳定；
- VCReg 是否比 CUT 更低或更平滑；
- 低距离是否伴随 feature norm 或有效秩异常下降。

## 6. 敏感度与汇总指标

### 6.1 小扰动敏感度

以 `weak` 相对 `clean` 的距离作为最小扰动敏感度：

$$
S_l^{\mathrm{cos}}=d_l^{\mathrm{cos}}(x,x_{\mathrm{weak}})
$$

如果需要按图像变化量归一化，必须同时报告图像空间扰动大小：

$$
S_l^{\mathrm{relative}}=
\frac{d_l^{\mathrm{cos}}(x,x')}
{D_{\mathrm{image}}(x,x')+\epsilon}
$$

第一版至少记录 normalized pixel L2；LPIPS/SSIM 可作为后续补充，不能在未实现时写入主结果。

### 6.2 稳定性汇总

每个模型、层和扰动条件至少输出：

- mean cosine distance；
- std cosine distance；
- P05/P50/P95 cosine distance；
- mean normalized L2；
- feature norm ratio；
- weak/medium/strong 的响应斜率；
- 随机场实例间均值和标准差。

### 6.3 不变性与可分性边界

本阶段的主结论只能写为“对所测扰动的稳定性”。距离更小不自动等于表示更好。

如果要声称没有 collapse，至少要检查：

- 原始 feature 的通道标准差；
- feature covariance 的 effective rank；
- 不同原图之间的同层 feature cosine 距离；
- clean 图像的内容差异没有被整体压平。

这些是本阶段的防御性统计，域 probe 和 paired retrieval 留到第二阶段。

## 7. 可复现性要求

每次实验必须保存 `manifest.json`，至少包含：

```json
{
  "seed": 1407,
  "dataset": "datasets/RRSHID/testA",
  "experiment": "cut_baseline",
  "checkpoint": "results/ckpt/cut_baseline/best_test_psnr.pt",
  "image_count": 304,
  "encoder_layers": ["stem", "downsample_1", "downsample_2", "enc_resblock_1", "enc_resblock_6"],
  "asm_parameters": {
    "field_type": "low_frequency_random",
    "field_seed": 1407,
    "d_range": [0.0, 1.0],
    "beta_range": [0.0, 1.0],
    "beta_scale": 0.5,
    "coupling": 0.0,
    "atmospheric_light": [1.0, 1.0, 1.0]
  },
  "perturbation_parameters": {},
  "preprocessing": {
    "normalization": "pixel / 127.5 - 1.0",
    "padding": "reflect to multiple of 4"
  }
}
```

每个生成的扰动样本必须能够由 `image_name + condition + seed + parameter_record` 重建。禁止只保存汇总数字而不保存扰动参数。

## 8. 输出文件

建议输出目录：

```text
results/feature_analysis/encoder_robustness/<experiment>/<dataset>/
```

至少包含：

```text
manifest.json
per_image_metrics.jsonl
summary.json
summary.md
curves.csv
```

`summary.md` 至少包含：

1. CUT 与 VCReg 的模型和 checkpoint；
2. 每个 Encoder 层的强度响应表；
3. 不同随机场实例和随机场统计量对比；
4. blur/gamma/组合扰动对比；
5. cosine 与 normalized L2 的并列结果；
6. VCReg-CUT 的绝对差值和相对变化；
7. feature norm、effective rank 和 collapse 风险提示。

## 9. 验收标准

第一阶段实现完成的最低标准：

- CUT 与 VCReg 能使用同一批扰动输入完成推理；
- 不使用 Expander 也能输出所有指定 Encoder 层的距离；
- 同位置 patch 的 cosine 和 normalized L2 计算通过人工小例子校验；
- `clean` 与 `weak` 条件均可复现；
- 所有 ASM 随机场、种子和强度参数写入 manifest；
- 随机场尺寸不匹配、数值范围非法或输出 transmission 越界时明确报错；
- 输出单图结果、汇总结果和 Markdown 报告；
- 至少有一个小样本 smoke test 和一个全量 304 图像测试；
- 结果中明确区分“Encoder 原始空间距离”和历史 PatchNCE Expander 距离。

## 10. 后续决策规则

### 支持 VCReg 改善鲁棒性的证据

同时观察到：

- VCReg 在 weak/medium ASM 扰动下 cosine 距离更低；
- 距离随强度变化更平滑；
- 随机场实例间方差更低；
- feature norm 和 effective rank 没有明显坍缩；
- clean 图像之间的内容差异仍被保留。

### 需要谨慎解释的结果

- 只有 cosine 变小而 normalized L2 变大；
- 只有某个随机实例或某个层变好；
- 所有图像之间的 feature 距离同时下降；
- effective rank 或通道标准差明显下降；
- ASM 随机场、强度或参数在两组模型之间不一致。

这些情况只能说明某种几何统计发生变化，不能直接称为“学习了更深层语义”。