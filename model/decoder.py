import torch.nn as nn

from .backbone import BlurUpsample, ResnetBlock, get_norm_layer


class Decoder(nn.Module):
    """Image decoder with configurable ResNet blocks and upsampling stages."""

    def __init__(self, bottleneck_channels=256, output_nc=3, ngf=64,
                 n_blocks=0, norm_type='instance', use_dropout=False,
                 upsample_mode='nearest'):
        super().__init__()
        if n_blocks < 0:
            raise ValueError('n_blocks must be non-negative')
        norm_layer = get_norm_layer(norm_type)
        if upsample_mode not in {'nearest', 'bilinear', 'antialias', 'transposed'}:
            raise ValueError(f'Unsupported upsample mode: {upsample_mode}')
        channels = bottleneck_channels
        self.resblocks = nn.ModuleList([
            ResnetBlock(channels, norm_type, use_dropout)
            for _ in range(n_blocks)
        ])
        self.upsample = nn.ModuleList()
        for _ in range(2):
            next_channels = channels // 2
            if next_channels < ngf:
                raise ValueError('bottleneck_channels is too small for decoder')
            if upsample_mode == 'transposed':
                layers: list[nn.Module] = [nn.ConvTranspose2d(
                    channels, next_channels, 3, 2, 1, output_padding=1)]
            else:
                if upsample_mode == 'antialias':
                    layers = [BlurUpsample(channels)]
                else:
                    mode = 'nearest' if upsample_mode == 'nearest' else 'bilinear'
                    layers = [nn.Upsample(scale_factor=2, mode=mode,
                                          align_corners=False if mode == 'bilinear' else None)]
                layers.append(nn.Conv2d(channels, next_channels, kernel_size=3, padding=1))
            layers.extend([norm_layer(next_channels), nn.ReLU(inplace=True)])
            self.upsample.append(nn.Sequential(*layers))
            channels = next_channels

        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        for block in self.resblocks:
            x = block(x)
        for layer in self.upsample:
            x = layer(x)
        return self.output(x)