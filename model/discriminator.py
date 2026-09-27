import torch.nn as nn

from .backbone import BlurDownsample, get_norm_layer


class PatchDiscriminator(nn.Module):
    """Fully convolutional PatchGAN discriminator returning logits."""

    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_type='instance',
                 downsample_mode='stride'):
        super().__init__()
        if n_layers < 1:
            raise ValueError('n_layers must be positive')
        norm_layer = get_norm_layer(norm_type)
        if downsample_mode not in {'stride', 'avgpool', 'antialias'}:
            raise ValueError(f'Unsupported downsample mode: {downsample_mode}')
        layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4,
                      stride=2 if downsample_mode == 'stride' else 1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        if downsample_mode == 'avgpool':
            layers.append(nn.AvgPool2d(2, stride=2))
        elif downsample_mode == 'antialias':
            layers.append(BlurDownsample(ndf))
        channels = ndf
        for layer_index in range(1, n_layers):
            next_channels = min(ndf * (2 ** layer_index), ndf * 8)
            layers.extend([
                nn.Conv2d(channels, next_channels, kernel_size=4,
                          stride=2 if downsample_mode == 'stride' else 1, padding=1),
                norm_layer(next_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            if downsample_mode == 'avgpool':
                layers.append(nn.AvgPool2d(2, stride=2))
            elif downsample_mode == 'antialias':
                layers.append(BlurDownsample(next_channels))
            channels = next_channels
        layers.extend([
            nn.Conv2d(channels, min(channels * 2, ndf * 8), kernel_size=4,
                      stride=1, padding=1),
            norm_layer(min(channels * 2, ndf * 8)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(min(channels * 2, ndf * 8), 1, kernel_size=4,
                      stride=1, padding=1),
        ])
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)