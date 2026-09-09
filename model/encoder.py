import torch
import torch.nn as nn

from .backbone import ResnetBlock, get_norm_layer


class Encoder(nn.Module):
    """CUT-style image encoder with two downsampling stages and ResNet blocks."""

    def __init__(self, input_nc=3, ngf=64, n_blocks=8,
                 norm_type='instance', use_dropout=False):
        super().__init__()
        if n_blocks < 0:
            raise ValueError('n_blocks must be non-negative')

        norm_layer = get_norm_layer(norm_type)
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
        )
        self.downsample = nn.ModuleList()
        channels = ngf
        for _ in range(2):
            next_channels = channels * 2
            self.downsample.append(nn.Sequential(
                nn.Conv2d(channels, next_channels, kernel_size=3,
                          stride=2, padding=1),
                norm_layer(next_channels),
                nn.ReLU(inplace=True),
            ))
            channels = next_channels

        self.resblocks = nn.ModuleList([
            ResnetBlock(channels, norm_type, use_dropout)
            for _ in range(n_blocks)
        ])
        self.output_channels = channels

    def forward(self, x, return_features=False):
        features = {}
        x = self.stem(x)
        features['stem'] = x
        for index, layer in enumerate(self.downsample, start=1):
            x = layer(x)
            features[f'downsample_{index}'] = x
        for index, block in enumerate(self.resblocks, start=1):
            x = block(x)
            features[f'resblock_{index}'] = x
        if return_features:
            return x, features
        return x