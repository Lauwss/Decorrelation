## Plan: 模块化 CUT Baseline

将生成器拆成独立的 Encoder、Decoder，并由 `network.py` 作为主网络统一组合。判别器单独放置，后续 PatchNCE、损失和训练逻辑分别维护，方便替换模块和做结构优化。

**目标结构**

```text
Input RGB
   |
Encoder
   |-- 7x7 Conv
   |-- Downsample 1
   |-- Downsample 2
   |-- 8 x ResBlock
   |
Bottleneck Feature
   |
Decoder
   |-- Upsample 1 + Conv
   |-- Upsample 2 + Conv
   |-- 7x7 Conv + Tanh
   |
Output RGB
```

**步骤**

1. **拆分基础组件**
   - [model/backbone.py](model/backbone.py) 只保留共享模块：
     - `ResnetBlock`
     - 归一化层构造函数
     - 卷积辅助函数
   - 不再在 `backbone.py` 中直接堆叠完整生成器。

2. **实现 Encoder**
   - 新增 [model/encoder.py](model/encoder.py)。
   - 定义 `Encoder` 类。
   - 结构包括：
     - `7x7 Conv`
     - 两次 `stride=2` 下采样
     - 8 个 ResBlock
   - 输入为 `[N, 3, H, W]`。
   - 输出 bottleneck feature，空间尺寸约为 `[H/4, W/4]`。
   - 提供 feature extraction 接口，供 PatchNCE 使用。

3. **实现 Decoder**
   - 新增 [model/decoder.py](model/decoder.py)。
   - 定义 `Decoder` 类。
   - 接收 Encoder 输出的 bottleneck feature。
   - 采用：
     - `nearest Upsample(scale_factor=2)`
     - `3x3 Conv`
     - 重复两次
     - `7x7 Conv + Tanh`
   - 输出恢复为 `[N, 3, H, W]`。

4. **实现主网络**
   - 新增 [model/network.py](model/network.py)。
   - 定义 `CUTGenerator` 或 `GeneratorNetwork`。
   - 内部组合：
     - `self.encoder`
     - `self.decoder`
   - 普通调用：
     - `output = network(input)`
   - 特征调用：
     - 返回 Encoder 和指定 ResBlock 的中间 feature。
   - 训练代码只依赖 `network.py`，不直接耦合 Encoder 和 Decoder 的内部实现。

5. **实现独立判别器**
   - 新增 [model/discriminator.py](model/discriminator.py)。
   - 定义 PatchGAN/70×70 PatchGAN。
   - 输出局部真实性 logits。
   - 不在判别器内部添加 sigmoid，交由 loss 处理。

6. **补齐 CUT 组件**
   - 新增 [model/patch_sample.py](model/patch_sample.py)：
     - 从 Network 返回的多层 feature 中抽样 patch
     - L2 normalization
     - MLP projection
   - 扩展 [utils/loss.py](utils/loss.py)：
     - GAN loss
     - PatchNCE loss
     - identity NCE
     - 多层特征损失聚合

7. **统一模型导出**
   - 可新增 [model/__init__.py](model/__init__.py)，集中导出：
     - `Encoder`
     - `Decoder`
     - `CUTGenerator`
     - `PatchDiscriminator`
     - `ResnetBlock`
   - 训练入口只从 `model` 包导入主接口，后续更换实现时不需要修改大量训练代码。

**推荐的构造接口**

- `Encoder(input_nc=3, ngf=64, n_blocks=8, norm_type="instance")`
- `Decoder(output_nc=3, ngf=64, n_blocks=8, norm_type="instance")`
- `CUTGenerator(input_nc=3, output_nc=3, ngf=64, n_blocks=8)`
- `PatchDiscriminator(input_nc=3, ndf=64, n_layers=3)`

其中 Encoder 和 Decoder 的通道约定保持一致：

```text
Encoder:
3 -> 64 -> 128 -> 256

Decoder:
256 -> 128 -> 64 -> 3
```

**模块职责边界**

- `backbone.py`：共享基础层和 ResBlock。
- `encoder.py`：编码、下采样、中间残差特征。
- `decoder.py`：解码、上采样、图像重建。
- `network.py`：组合 Encoder/Decoder，并提供统一 forward 和 feature 接口。
- `discriminator.py`：PatchGAN 判别。
- `patch_sample.py`：PatchNCE 所需的 patch 抽取与投影。
- `utils/loss.py`：损失函数，不包含网络结构。
- `train.py`：训练流程、优化器和 checkpoint，不直接实现模型细节。

**验证顺序**

1. 单独验证 Encoder：
   - 输入 `[2, 3, 128, 128]`
   - 输出 `[2, 256, 32, 32]`
2. 单独验证 Decoder：
   - 输入 `[2, 256, 32, 32]`
   - 输出 `[2, 3, 128, 128]`
3. 验证 `network.py` 的端到端输出尺寸。
4. 统计网络结构：
   - 两次下采样
   - 8 个 ResBlock
   - 两次上采样
5. 执行一次反向传播，确认 Encoder、Decoder 参数都有梯度。
6. 验证 Discriminator 输出 Patch logits。
7. 验证 PatchSample 和 PatchNCE feature shape。
8. 使用真实 RRSHID 数据执行一个 batch 的完整前向和反向传播。

**范围决定**

- 仍然采用 RGB `3 -> 3`。
- 仍然采用非配对 CUT 训练。
- 默认使用 `Upsample + Conv2d`，避免反卷积棋盘格伪影。
- 默认使用 8 个 ResBlock。
- `network.py` 是训练代码面对的主生成器接口。
- Encoder、Decoder 和 Discriminator 均独立成文件，便于后续替换和优化。
- 本轮不直接修改代码，等待你确认这个模块拆分方案后再开始实现。
