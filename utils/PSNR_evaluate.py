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
def evaluate_paired_psnr(generator, source_dir, target_dir, device, max_images=0):
	"""Evaluate a generator against same-named images in source and target folders."""
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
	scores = []
	for name in paired_names:
		try:
			source = _load_image(source_paths[name]).unsqueeze(0).to(device)
			target = _load_image(target_paths[name]).unsqueeze(0).to(device)
		except (OSError, ValueError):
			continue
		source, height, width = _pad_to_multiple(source)
		output = generator(source)[:, :, :height, :width]
		if output.shape != target.shape:
			continue
		scores.append(calculate_psnr(output, target))
	generator.train(was_training)
	if not scores:
		raise RuntimeError('No valid same-sized image pairs were available for PSNR')
	return sum(scores) / len(scores)
