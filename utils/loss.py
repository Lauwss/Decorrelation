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


class ExpanderDecorrelationLoss(nn.Module):
	"""VICReg-style channel decorrelation for projected patch features."""

	def __init__(self, min_std=0.5, eps=1e-6):
		super().__init__()
		self.min_std = min_std
		self.eps = eps

	def _statistics(self, projected):
		if projected.ndim != 2:
			raise ValueError('projected features must have shape [samples, channels]')
		if projected.shape[0] < 2:
			zero = projected.float().sum() * 0.0
			return zero, zero, {
				'mean_abs_corr': zero.detach(),
				'offdiag_energy': zero.detach(),
				'mean_std': zero.detach(),
				'min_std': zero.detach(),
				'effective_rank': zero.detach(),
				'pc1_ratio': zero.detach(),
			}

		with torch.autocast(
			device_type=projected.device.type, enabled=False):
			values = projected.float()
			centered = values - values.mean(dim=0, keepdim=True)
			covariance = centered.transpose(0, 1).matmul(centered)
			covariance = covariance / (values.shape[0] - 1)
			channels = values.shape[1]
			diagonal = torch.diagonal(covariance)
			off_diagonal = covariance - torch.diag_embed(diagonal)
			covariance_loss = off_diagonal.square().sum() / max(
				channels * (channels - 1), 1)
			std = torch.sqrt(diagonal.clamp_min(0.0) + self.eps)
			variance_loss = F.relu(self.min_std - std).square().mean()

			correlation = covariance / (std[:, None] * std[None, :] + self.eps)
			off_diagonal_correlation = correlation - torch.diag_embed(
				torch.diagonal(correlation))
			offdiag_count = max(channels * (channels - 1), 1)
			mean_abs_corr = off_diagonal_correlation.abs().sum() / offdiag_count
			offdiag_energy = off_diagonal.square().sum() / (
				diagonal.square().sum() + self.eps)

			eigenvalues = torch.linalg.eigvalsh(
				covariance.detach()).clamp_min(0.0)
			eigenvalue_sum = eigenvalues.sum() + self.eps
			probabilities = eigenvalues / eigenvalue_sum
			effective_rank = torch.exp(-(
				probabilities * torch.log(probabilities + self.eps)).sum())
			pc1_ratio = eigenvalues[-1] / eigenvalue_sum
		stats = {
			'mean_abs_corr': mean_abs_corr.detach(),
			'offdiag_energy': offdiag_energy.detach(),
			'mean_std': std.mean().detach(),
			'min_std': std.min().detach(),
			'effective_rank': effective_rank.detach(),
			'pc1_ratio': pc1_ratio.detach(),
		}
		return covariance_loss, variance_loss, stats

	def forward(self, fake_projected, real_projected):
		if len(fake_projected) != len(real_projected):
			raise ValueError('fake and real projected layer counts must match')
		if not fake_projected:
			raise ValueError('at least one projected feature layer is required')
		covariance_losses = []
		variance_losses = []
		statistics = []
		for fake, real in zip(fake_projected, real_projected):
			fake_cov, fake_var, fake_stats = self._statistics(fake)
			real_cov, real_var, real_stats = self._statistics(real)
			covariance_losses.append((fake_cov + real_cov) * 0.5)
			variance_losses.append((fake_var + real_var) * 0.5)
			statistics.append({
				name: (fake_stats[name] + real_stats[name]) * 0.5
				for name in fake_stats
			})
		covariance_loss = torch.stack(covariance_losses).mean()
		variance_loss = torch.stack(variance_losses).mean()
		stats = {
			name: torch.stack([layer[name] for layer in statistics]).mean()
			for name in statistics[0]
		}
		return covariance_loss, variance_loss, stats

class IdentityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, input, target):
        return self.loss(input, target)