from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def calculate_psnr(prediction, target, eps=1e-10):
	"""Return mean PSNR for image tensors normalized to [-1, 1]."""
	if prediction.shape != target.shape:
		raise ValueError('prediction and target must have the same shape')
	prediction = ((prediction.detach().float() + 1.0) * 0.5).clamp(0.0, 1.0)
	target = ((target.detach().float() + 1.0) * 0.5).clamp(0.0, 1.0)
	mse = (prediction - target).square().flatten(1).mean(dim=1)
	return (-10.0 * torch.log10(mse.clamp_min(eps))).mean().item()


def _load_image(path):
	image = Image.open(path).convert('RGB')
	return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0


def _pad_to_multiple(image, multiple=4):
	_, _, height, width = image.shape
	pad_height = (multiple - height % multiple) % multiple
	pad_width = (multiple - width % multiple) % multiple
	if not pad_height and not pad_width:
		return image, height, width
	return F.pad(image, (0, pad_width, 0, pad_height), mode='reflect'), height, width


@torch.no_grad()
def evaluate_paired_psnr(generator, source_dir, target_dir, device, max_images=0,
						 batch_size=1):
	"""Evaluate a generator against same-named images in source and target folders."""
	if batch_size < 1:
		raise ValueError('batch_size must be positive')
	source_paths = {path.name: path for path in Path(source_dir).glob('*.png')}
	target_paths = {path.name: path for path in Path(target_dir).glob('*.png')}
	paired_names = sorted(source_paths.keys() & target_paths.keys())
	if max_images:
		paired_names = paired_names[:max_images]
	if not paired_names:
		raise FileNotFoundError(
			f'No same-named PNG pairs found in {source_dir} and {target_dir}'
		)

	was_training = generator.training
	generator.eval()
	score_sum = 0.0
	score_count = 0
	sources = []
	targets = []

	def evaluate_batch():
		nonlocal score_sum, score_count
		if not sources:
			return
		source_batch = torch.stack(sources).to(device)
		target_batch = torch.stack(targets).to(device)
		height, width = target_batch.shape[-2:]
		output = generator(source_batch)[:, :, :height, :width]
		batch_count = source_batch.shape[0]
		score_sum += calculate_psnr(output, target_batch) * batch_count
		score_count += batch_count
		sources.clear()
		targets.clear()

	for name in paired_names:
		try:
			source = _load_image(source_paths[name])
			target = _load_image(target_paths[name])
		except (OSError, ValueError):
			continue
		if source.shape != target.shape:
			continue
		if targets and target.shape != targets[0].shape:
			evaluate_batch()
		source, _, _ = _pad_to_multiple(source.unsqueeze(0))
		source = source.squeeze(0)
		sources.append(source)
		targets.append(target)
		if len(sources) == batch_size:
			evaluate_batch()
	evaluate_batch()
	generator.train(was_training)
	if not score_count:
		raise RuntimeError('No valid same-sized image pairs were available for PSNR')
	return score_sum / score_count
