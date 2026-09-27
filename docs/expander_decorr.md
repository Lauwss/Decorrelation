# Expander 输出空间去相关：实现与实验规范

## 1. 目标与边界

本实验使用 VICReg 风格的二阶统计约束，降低 CUT 的 Expander 最终输出中不同通道之间的线性相关性。当前阶段只研究通道级 decorrelation 对训练和表示统计的影响。

主设置：

```text
统计对象：Expander 最终 Linear 输出，L2 normalize 之前
训练分支：fake_B + real_A
默认层：enc_resblock_6
统计样本：当前 batch 的 B × P 个 patch
队列：关闭
PatchNCE：继续使用 L2-normalized Expander 输出
```

`fake-only`、first hidden 和 normalized output 保留为消融，不作为主方案。后续重点是验证不同测量位置对跨域距离、域可分性和跨退化稳定性的影响。

## 2. 当前特征路径

当前 `PatchSampleF` 对每个选中的 Encoder feature map 执行：

```text
feature map [B, C, H, W]
    -> patch sampling
    -> sampled [B × P, C]
    -> Linear(C, 256)
    -> ReLU
    -> Linear(256, 256)
    -> projected [B × P, 256]
    -> L2 normalize
    -> embedding [B × P, 256]
```

设：

$$
z=W_2\operatorname{ReLU}(W_1x+b_1)+b_2,
$$

$$
\hat z=\frac{z}{\lVert z\rVert_2}.
$$

对去相关损失使用 `projected`，对 PatchNCE 使用 `embedding`。不应把主 VICReg variance loss 直接施加在 $\hat z$ 上，因为逐样本 L2 normalization 固定了样本范数，会改变通道方差的解释。

## 3. 为什么约束最终 projected output

如果只约束 first hidden $h$：

$$
h=\operatorname{ReLU}(W_1x+b_1),
$$

第二层仍然会重新组合通道：

$$
z=W_2h+b_2,
$$

因此：

$$
\operatorname{Cov}(z)=W_2\operatorname{Cov}(h)W_2^{\mathsf T}.
$$

即使 $\operatorname{Cov}(h)$ 接近对角矩阵，$W_2$ 仍可能重新产生高度相关的输出。目标是改善最终 Expander 表示空间，因此主损失必须约束 $z$。first hidden 仅作为低干预强度的对照实验。

## 4. 为什么主方案使用 fake + real_A

Expander 是共享投影器，同时处理 `real_A`、`fake_B` 以及可选的 identity 分支。若只约束 `fake_B`，得到的是：

$$
\mathcal L_{\mathrm{decorr}}=\mathcal L_{\mathrm{decorr}}(z_{\mathrm{fake}}),
$$

它只能说明生成分支是否被修正，不能说明共享 Expander 空间是否形成一致的去相关坐标系。

主方案使用：

$$
\mathcal L_{\mathrm{decorr}}^{\mathrm{total}}
=
\frac{1}{2}
\left[
\mathcal L_{\mathrm{decorr}}(z_{\mathrm{fake}})
+
\mathcal L_{\mathrm{decorr}}(z_{\mathrm{real\_A}})
\right].
$$

`real_A` 的 embedding 作为 PatchNCE key 时仍然 detach，但 `real_A` 的 `projected` 必须保留梯度：

```text
real embedding -> detach -> PatchNCE key
real projected  -> keep grad -> decorrelation
```

不要用覆盖整个 real 分支的 `torch.no_grad()`，否则 real decorrelation 不会更新 Encoder 或 Expander。

## 5. VICReg 风格损失

对每个 Encoder 层单独计算。设某层输出为：

$$
Z_l\in\mathbb{R}^{N\times d},
\qquad N=B\times P,
\qquad d=256.
$$

不同层的 Expander 参数和语义不同，不能把不同层直接拼接后只计算一个协方差矩阵。

### 5.1 中心化和协方差

$$
\mu_l=\frac{1}{N}\sum_{n=1}^{N}Z_{l,n},
$$

$$
\widetilde Z_l=Z_l-\mu_l,
$$

$$
C_l=\frac{\widetilde Z_l^{\mathsf T}\widetilde Z_l}{N-1}.
$$

协方差计算在 float32 中进行，即使主训练启用了 AMP。若 $N<2$，该层返回零损失并跳过统计。

### 5.2 非对角协方差损失

$$
\mathcal L_{\mathrm{cov}}^{(l)}
=
\frac{1}{d(d-1)}
\sum_{i\neq j}C_{l,ij}^{2}.
$$

### 5.3 方差下界

$$
\sigma_{l,i}=\sqrt{C_{l,ii}+\varepsilon},
$$

$$
\mathcal L_{\mathrm{var}}^{(l)}
=
\frac{1}{d}\sum_{i=1}^{d}
\max\left(0,\sigma_{\min}-\sigma_{l,i}\right)^2.
$$

方差项防止所有 projected 通道缩小，第一版必须弱于 covariance 项。

### 5.4 多层和多分支聚合

对选定层集合 $S$ 逐层计算再取平均：

$$
\mathcal L_{\mathrm{cov}}
=
\frac{1}{|S|}\sum_{l\in S}
\frac{1}{2}
\left[
\mathcal L_{\mathrm{cov}}(Z_{l,\mathrm{fake}})
+
\mathcal L_{\mathrm{cov}}(Z_{l,\mathrm{real\_A}})
\right].
$$

$$
\mathcal L_{\mathrm{var}}
=
\frac{1}{|S|}\sum_{l\in S}
\frac{1}{2}
\left[
\mathcal L_{\mathrm{var}}(Z_{l,\mathrm{fake}})
+
\mathcal L_{\mathrm{var}}(Z_{l,\mathrm{real\_A}})
\right].
$$

$$
\mathcal L_{\mathrm{decorr}}
=
\lambda_{\mathrm{cov}}(e)\mathcal L_{\mathrm{cov}}
+
\lambda_{\mathrm{var}}(e)\mathcal L_{\mathrm{var}}.
$$

## 6. 训练目标和梯度边界

$$
\mathcal L_G'
=
\lambda_G\mathcal L_{\mathrm{GAN}}
+
\lambda_{\mathrm{NCE}}\mathcal L_{\mathrm{NCE}}
+
\mathcal L_{\mathrm{decorr}}.
$$

只进行一次反向传播：

```python
loss_g = loss_g_gan + loss_g_nce + loss_expander
scaler.scale(loss_g).backward()
scaler.step(optimizer_g)
scaler.step(optimizer_f)
```

梯度边界：

```text
fake projected：有梯度，更新 Generator Encoder 和 Expander
real projected：有梯度，更新 Generator Encoder 和 Expander
real embedding：作为 NCE key 时 detach
discriminator：不接收 decorrelation 梯度
```

## 7. `PatchSampleF` 接口改造

`model/feature.py` 的 `forward()` 增加：

```python
def forward(
    self,
    features,
    num_patches=256,
    patch_ids=None,
    return_projected=False,
):
```

默认 `return_projected=False` 时保持原有两项返回值。开启时返回：

```python
embeddings, patch_ids, projected_features
```

其中：

```text
embeddings: [B × P, 256]，L2 normalized
patch_ids: 每层的空间采样索引
projected_features: [B × P, 256]，normalize 前的 Expander 输出
```

推荐将当前 `Sequential` 改成明确的 `ModuleDict`：

```python
nn.ModuleDict({
    'input': nn.Linear(input_channels, self.nc),
    'activation': nn.ReLU(inplace=True),
    'output': nn.Linear(self.nc, self.nc),
})
```

主方案只暴露 `projected`，first hidden 仅在位置消融中额外暴露。

## 8. 训练步骤

生成器更新阶段：

1. 计算 `fake_b` 的 Encoder feature。
2. 从 fake feature 采样，得到 `fake_embeddings`、`patch_ids` 和 `fake_projected`。
3. 使用相同 `patch_ids` 从 `real_A` feature 采样，得到 `real_embeddings` 和 `real_projected`。
4. 对 `real_embeddings` detach，仅用于 PatchNCE key。
5. 对 fake 和 real projected 逐层计算 covariance/variance。
6. 将 decorrelation loss 加入生成器总损失。
7. 一次 backward，同时更新 Generator 和 Expander。

示意代码：

```python
fake_embeddings, patch_ids, fake_projected = feature_net(
    fake_feature_list,
    num_patches=num_patches,
    return_projected=True,
)
real_embeddings, _, real_projected = feature_net(
    real_feature_list,
    num_patches=num_patches,
    patch_ids=patch_ids,
    return_projected=True,
)
real_keys = [embedding.detach() for embedding in real_embeddings]
loss_nce = nce_loss(fake_embeddings, real_keys, num_patches)
loss_cov, loss_var, stats = decorr_loss(
    fake_projected,
    real_projected,
)
loss_expander = lambda_cov * loss_cov + lambda_var * loss_var
```

identity NCE 第一版不加入 decorrelation，待主实验稳定后再测试 `fake + real_A + idt_B`。

## 9. Warmup 和初始参数

$$
 r(e)=
\begin{cases}
1, & E_w=0,\\
\min\left(1,\frac{e}{E_w}\right), & E_w>0.
\end{cases}
$$

$$
\lambda_{\mathrm{cov}}(e)=r(e)\lambda_{\mathrm{cov}}^{\mathrm{target}},
\qquad
\lambda_{\mathrm{var}}(e)=r(e)\lambda_{\mathrm{var}}^{\mathrm{target}}.
$$

推荐初始配置：

```text
use_expander_decorr = True
expander_decorr_layers = enc_resblock_6
expander_decorr_branches = fake,real_a
lambda_expander_decorr = 1e-6
lambda_expander_decorr_var = 1e-7
expander_decorr_min_std = 0.5
expander_decorr_warmup_epochs = 10
queue = disabled
```

命令行参数：

```text
--use-expander-decorr
--lambda-expander-decorr
--lambda-expander-decorr-var
--expander-decorr-layers
--expander-decorr-branches
--expander-decorr-min-std
--expander-decorr-warmup-epochs
```

层名必须属于当前 PatchNCE 选中的层，非法层名启动时直接报错。

## 10. 不使用 queue 的理由

当前默认配置下：

```text
batch_size = 16
num_patches = 256
N = 4096
projected dimension = 256
```

当前 batch 已经足以估计 256 维协方差。Expander 输出分布会随 Generator 和 Expander 参数变化，加入历史 queue 会引入滞后统计和额外状态。

只有在 batch 或 patch 数量显著降低、协方差指标波动过大时才考虑 queue。若未来加入 queue，只保存 `detach().float()` 的历史 projected，且 queue 永不回传梯度。

## 11. 日志指标

训练 CSV 至少增加：

```text
loss_expander_cov
loss_expander_var
expander_mean_abs_corr
expander_offdiag_energy
expander_mean_std
expander_min_std
expander_effective_rank
expander_pc1_ratio
expander_decorr_weight
```

相关系数矩阵：

$$
R_{ij}=\frac{C_{ij}}{\sigma_i\sigma_j+\varepsilon}.
$$

平均绝对相关系数：

$$
\operatorname{mean\_abs\_corr}
=
\frac{1}{d(d-1)}\sum_{i\neq j}|R_{ij}|.
$$

非对角能量占比：

$$
\operatorname{offdiag\_energy}
=
\frac{\sum_{i\neq j}C_{ij}^{2}}
{\sum_i C_{ii}^{2}+\varepsilon}.
$$

有效秩：

$$
 r_{\mathrm{eff}}
=
\exp\left(-\sum_i p_i\log(p_i+\varepsilon)\right),
\qquad
p_i=\frac{\lambda_i}{\sum_j\lambda_j+\varepsilon}.
$$

若 `offdiag_energy` 下降但 `effective_rank` 也下降、`pc1_ratio` 上升，更可能是表示塌缩，而不是健康的通道去相关。

## 12. 当前阶段的评估

当前阶段不把 generated-clear cosine distance 或退化敏感性单独解释为域不变性证据。训练和离线分析至少报告：

```text
paired PSNR
identity PSNR
decoder oracle gap
PatchNCE loss
GAN loss
source -> generated distance
clear -> generated distance
source -> clear distance
Expander off-diagonal energy
Expander effective rank
Expander PC1 explained ratio
```

图像质量指标用于判断去相关是否损害主任务；它们不是跨域不变性的直接度量。

## 13. 后续机理分析（不属于当前实现）

如果后续需要研究内容因素 $A$、退化因素 $t$、$\operatorname{Cov}(A,t)$ 或输出方差交叉项，应先明确可观测的因素表示和配对数据构造，再单独设计实验。当前通道级 covariance loss 不应直接命名为 $\operatorname{Cov}(A,t)$ loss，也不应据此宣称已经获得完整的跨域不变性。

当前实现只优化：

$$
\operatorname{Cov}(z_i,z_j)\approx 0,
\qquad i\neq j.
$$

是否形成可迁移的域不变表征，以及它是否影响输出方差，需要后续机理分析验证，不纳入本阶段的单一结论。

## 14. 实验矩阵

### 14.1 分支消融

固定 projected output、`enc_resblock_6` 和 `lambda_cov=1e-6`：

| 实验 | 分支 |
| --- | --- |
| D0 | 无 decorrelation |
| D1 | fake |
| D2 | fake + real_A |
| D3 | fake + real_A + idt_B |

主比较是 D1 与 D2，用于判断共享表示空间约束是否优于只修正生成分支。

### 14.2 位置消融

固定 `fake + real_A` 和最深层：

| 实验 | 位置 |
| --- | --- |
| P0 | 无 decorrelation |
| P1 | first hidden |
| P2 | final projected output |
| P3 | first hidden + final projected |

主方案为 P2。

### 14.3 强度和层数消融

固定 `fake + real_A` 和 projected output：

| 实验 | 层 | covariance weight |
| --- | --- | ---: |
| S0 | 无 | 0 |
| S1 | `enc_resblock_6` | `1e-6` |
| S2 | `enc_resblock_6` | `3e-6` |
| S3 | `enc_resblock_6` | `1e-5` |
| S4 | `enc_resblock_1,enc_resblock_6` | `3e-6` |
| S5 | 全部 selected layers | `1e-6` |

所有消融必须固定 batch size、patch 数、数据划分、随机种子和其他训练参数。

## 15. 实现顺序与验证

按以下顺序实现：

1. 在 `model/feature.py` 中增加 `return_projected`，默认行为保持不变。
2. 在 `utils/loss.py` 中实现 `ExpanderDecorrelationLoss`。
3. 在 `train.py` 中接入 fake 和 real_A projected，并保持 real NCE key detached。
4. 增加参数解析、层名验证和 warmup。
5. 在 `utils/training.py` 中增加 CSV 指标。
6. 在 `analyze_features.py` 中增加 projected covariance、有效秩和相关性统计。

首先执行：

```bash
python -m py_compile model/feature.py train.py utils/loss.py utils/training.py analyze_features.py
```

最小验证：

```text
正则关闭：新损失和新指标为零，原有 PatchNCE 行为不变
正则开启：projected shape 为 [B × P, 256]
开启后：Generator Encoder 和 feature_net 都获得梯度
real NCE key：没有梯度
没有 NaN 或 Inf
单 epoch：CSV 和 checkpoint 正常写入
```

最终只有在以下条件大致同时满足时，才判定方案有效：

```text
offdiag covariance 下降
mean absolute correlation 下降
effective rank 不下降
PC1 explained ratio 不明显升高
PatchNCE 没有明显恶化
paired PSNR 不下降
identity PSNR 不下降
```

## 16. 明确不采用的方案

1. 不再实现 Encoder GAP decorrelation；该方案已从实验主线移除。
2. 不在 normalized embedding 上使用完整 VICReg variance floor。
3. 不把不同 Encoder 层的 projected 直接拼接计算一个 covariance。
4. 不把 real 分支整体放进 `torch.no_grad()` 后再计算 decorrelation。
5. 不在第一版引入 queue、momentum encoder 或额外 checkpoint 状态。
6. 不仅用 generated-clear cosine distance 作为成功标准。
