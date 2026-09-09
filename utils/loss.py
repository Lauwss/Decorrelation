"""
损失函数集合

-GAN_Loss: GAN 损失函数
-PathNCELoss: PatchNCE 损失函数
-Idt_Loss: Identity 损失函数
-Perceptual_Loss: 感知损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
	def __init__(self, gan_mode='lsgan'):
		super().__init__()
		if gan_mode == 'lsgan':
			self.loss = nn.MSELoss()
		elif gan_mode == 'vanilla':
			self.loss = nn.BCEWithLogitsLoss()
		else:
			raise ValueError(f'Unsupported GAN mode: {gan_mode}')
		self.gan_mode = gan_mode

	def forward(self, prediction, target_is_real):
		target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
		return self.loss(prediction, target)


class PatchNCELoss(nn.Module):
	def __init__(self, temperature=0.07):
		super().__init__()
		self.temperature = temperature

	def forward(self, query, key):
		if query.shape != key.shape:
			raise ValueError('query and key must have the same shape')
		if query.ndim != 3:
			raise ValueError('PatchNCE expects [batch, patches, channels] tensors')
		query = F.normalize(query, dim=-1)
		key = F.normalize(key.detach(), dim=-1)
		logits = torch.bmm(query, key.transpose(1, 2)) / self.temperature
		labels = torch.arange(query.shape[1], device=query.device)
		labels = labels.unsqueeze(0).expand(query.shape[0], -1).reshape(-1)
		return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels)


class MultiLayerPatchNCELoss(nn.Module):
	def __init__(self, temperature=0.07):
		super().__init__()
		self.patch_nce = PatchNCELoss(temperature)

	def forward(self, query_features, key_features, num_patches):
		losses = []
		for query, key in zip(query_features, key_features):
			if query.shape != key.shape:
				raise ValueError('query and key patch shapes must match')
			batch_size = query.shape[0] // num_patches
			query = query.reshape(batch_size, num_patches, -1)
			key = key.reshape(batch_size, num_patches, -1)
			losses.append(self.patch_nce(query, key))
		if not losses:
			raise ValueError('at least one feature layer is required')
		return torch.stack(losses).mean()

class IdentityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, input, target):
        return self.loss(input, target)