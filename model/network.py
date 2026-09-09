import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder


class CUTGenerator(nn.Module):
    """Composable CUT generator exposing encoder features for PatchNCE."""

    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=8,
                 norm_type='instance', use_dropout=False):
        super().__init__()
        self.encoder = Encoder(
            input_nc=input_nc,
            ngf=ngf,
            n_blocks=n_blocks,
            norm_type=norm_type,
            use_dropout=use_dropout,
        )
        self.decoder = Decoder(
            bottleneck_channels=self.encoder.output_channels,
            output_nc=output_nc,
            ngf=ngf,
            norm_type=norm_type,
        )

    def encode(self, x, return_features=False):
        return self.encoder(x, return_features=return_features)

    def decode(self, bottleneck):
        return self.decoder(bottleneck)

    def forward(self, x, return_features=False):
        bottleneck, features = self.encoder(x, return_features=True)
        output = self.decoder(bottleneck)
        if return_features:
            features['bottleneck'] = bottleneck
            return output, features
        return output


GeneratorNetwork = CUTGenerator