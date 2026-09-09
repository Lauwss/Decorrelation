import torch.nn as nn

from .backbone import get_norm_layer


class Decoder(nn.Module):
    """CUT-style decoder that reconstructs an RGB image from bottleneck features."""

    def __init__(self, bottleneck_channels=256, output_nc=3, ngf=64,
                 norm_type='instance'):
        super().__init__()
        norm_layer = get_norm_layer(norm_type)
        channels = bottleneck_channels
        self.upsample = nn.ModuleList()
        for _ in range(2):
            next_channels = channels // 2
            if next_channels < ngf:
                raise ValueError('bottleneck_channels is too small for decoder')
            self.upsample.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(channels, next_channels, kernel_size=3, padding=1),
                norm_layer(next_channels),
                nn.ReLU(inplace=True),
            ))
            channels = next_channels

        self.output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(channels, output_nc, kernel_size=7, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        for layer in self.upsample:
            x = layer(x)
        return self.output(x)