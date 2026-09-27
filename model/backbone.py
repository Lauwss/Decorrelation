import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm_layer(norm_type='instance'):
	if norm_type == 'instance':
		return lambda channels: nn.InstanceNorm2d(
			channels, affine=False, track_running_stats=False)
	if norm_type == 'batch':
		return lambda channels: nn.BatchNorm2d(channels)
	if norm_type == 'none':
		return lambda channels: nn.Identity()
	raise ValueError(f'Unsupported normalization: {norm_type}')


class BlurDownsample(nn.Module):
	def __init__(self, channels):
		super().__init__()
		kernel = torch.tensor([1., 3., 3., 1.])
		kernel = kernel[:, None] * kernel[None, :]
		kernel = kernel / kernel.sum()
		self.register_buffer('kernel', kernel[None, None].repeat(channels, 1, 1, 1))
		self.kernel: torch.Tensor

	def forward(self, x):
		return F.conv2d(x, self.kernel, stride=2, padding=1, groups=x.shape[1])


class BlurUpsample(nn.Module):
	def __init__(self, channels):
		super().__init__()
		kernel = torch.tensor([1., 3., 3., 1.])
		kernel = kernel[:, None] * kernel[None, :]
		kernel = kernel / kernel.sum()
		self.register_buffer('kernel', kernel[None, None].repeat(channels, 1, 1, 1))
		self.kernel: torch.Tensor

	def forward(self, x):
		x = F.interpolate(x, scale_factor=2, mode='nearest')
		x = F.pad(x, (0, 1, 0, 1))
		return F.conv2d(x, self.kernel, padding=1, groups=x.shape[1])


def initialize_weights(module, init_type='xavier', init_gain=0.02):
	if init_type not in {'normal', 'xavier', 'kaiming', 'orthogonal'}:
		raise ValueError(f'Unsupported initialization: {init_type}')
	def initialize(child):
		if isinstance(child, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
			if init_type == 'normal':
				nn.init.normal_(child.weight, 0.0, init_gain)
			elif init_type == 'xavier':
				nn.init.xavier_normal_(child.weight, gain=init_gain)
			elif init_type == 'kaiming':
				nn.init.kaiming_normal_(child.weight, a=0, mode='fan_in')
			else:
				nn.init.orthogonal_(child.weight, gain=init_gain)
			if child.bias is not None:
				nn.init.constant_(child.bias, 0.0)
		elif isinstance(child, (nn.BatchNorm2d, nn.InstanceNorm2d)) and child.affine:
			nn.init.normal_(child.weight, 1.0, init_gain)
			nn.init.constant_(child.bias, 0.0)
	module.apply(initialize)


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