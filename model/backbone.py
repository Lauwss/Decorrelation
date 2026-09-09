import torch
import torch.nn as nn


def get_norm_layer(norm_type='instance'):
	if norm_type == 'instance':
		return lambda channels: nn.InstanceNorm2d(
			channels, affine=False, track_running_stats=False)
	if norm_type == 'batch':
		return lambda channels: nn.BatchNorm2d(channels)
	if norm_type == 'none':
		return lambda channels: nn.Identity()
	raise ValueError(f'Unsupported normalization: {norm_type}')


class ResnetBlock(nn.Module):
	def __init__(self, channels, norm_type='instance', use_dropout=False):
		super().__init__()
		norm_layer = get_norm_layer(norm_type)
		layers = [
			nn.ReflectionPad2d(1),
			nn.Conv2d(channels, channels, kernel_size=3, bias=True),
			norm_layer(channels),
			nn.ReLU(inplace=True),
		]
		if use_dropout:
			layers.append(nn.Dropout(0.5))
		layers.extend([
			nn.ReflectionPad2d(1),
			nn.Conv2d(channels, channels, kernel_size=3, bias=True),
			norm_layer(channels),
		])
		self.block = nn.Sequential(*layers)

	def forward(self, x):
		return x + self.block(x)