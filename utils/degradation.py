"""Reproducible low-frequency random-field degradations for encoder analysis."""

from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CommonDegradationConfig:
    """Parameters for mild image-space nuisance perturbations."""

    blur_sigma: float = 0.0
    gamma: float = 1.0
    noise_std: float = 0.0
    contrast: float = 1.0
    seed: int = 1407

    def __post_init__(self):
        if self.blur_sigma < 0:
            raise ValueError('blur_sigma must be non-negative')
        if self.gamma <= 0:
            raise ValueError('gamma must be positive')
        if self.noise_std < 0:
            raise ValueError('noise_std must be non-negative')
        if self.contrast < 0:
            raise ValueError('contrast must be non-negative')

    def to_dict(self):
        return asdict(self)


def _as_batch(images):
    if images.ndim == 3:
        return images.unsqueeze(0), True
    if images.ndim == 4:
        return images, False
    raise ValueError('images must have shape [C, H, W] or [B, C, H, W]')


def _validate_images(images):
    if images.shape[1] != 3:
        raise ValueError('images must contain exactly three RGB channels')
    if not images.is_floating_point():
        raise TypeError('images must use a floating-point dtype')
    if images.amin() < -1.0001 or images.amax() > 1.0001:
        raise ValueError('images must be normalized to the [-1, 1] range')


def _gaussian_kernel(sigma, device, dtype):
    radius = max(1, int(3 * sigma + 0.5))
    coordinates = torch.arange(
        -radius, radius + 1, device=device, dtype=dtype)
    kernel_1d = torch.exp(-coordinates.square() / (2 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return kernel_2d


def apply_gaussian_blur(images, sigma):
    """Apply channel-wise Gaussian blur to [-1, 1] RGB tensors."""
    images, squeeze_batch = _as_batch(images)
    _validate_images(images)
    if sigma < 0:
        raise ValueError('sigma must be non-negative')
    if sigma == 0:
        return images.clone().squeeze(0) if squeeze_batch else images.clone()
    kernel = _gaussian_kernel(sigma, images.device, images.dtype)
    kernel = kernel.expand(3, 1, *kernel.shape)
    padding = kernel.shape[-1] // 2
    padded = F.pad(images, (padding, padding, padding, padding), mode='reflect')
    blurred = F.conv2d(padded, kernel, groups=3)
    return blurred.squeeze(0) if squeeze_batch else blurred


def apply_gamma(images, gamma):
    """Apply gamma correction in [0, 1] image space."""
    images, squeeze_batch = _as_batch(images)
    _validate_images(images)
    if gamma <= 0:
        raise ValueError('gamma must be positive')
    if gamma == 1:
        return images.clone().squeeze(0) if squeeze_batch else images.clone()
    image_zero_one = ((images + 1) / 2).clamp(0, 1)
    adjusted = image_zero_one.pow(gamma) * 2 - 1
    return adjusted.squeeze(0) if squeeze_batch else adjusted


def apply_gaussian_noise(images, std, seed=1407):
    """Add deterministic Gaussian noise in normalized image space."""
    images, squeeze_batch = _as_batch(images)
    _validate_images(images)
    if std < 0:
        raise ValueError('std must be non-negative')
    generator = torch.Generator(device=images.device)
    generator.manual_seed(seed)
    noise = torch.randn(
        images.shape, generator=generator, device=images.device,
        dtype=images.dtype) * std
    adjusted = (images + noise).clamp(-1, 1)
    return adjusted.squeeze(0) if squeeze_batch else adjusted


def apply_contrast(images, contrast):
    """Scale contrast around the midpoint in [0, 1] image space."""
    images, squeeze_batch = _as_batch(images)
    _validate_images(images)
    if contrast < 0:
        raise ValueError('contrast must be non-negative')
    if contrast == 1:
        return images.clone().squeeze(0) if squeeze_batch else images.clone()
    image_zero_one = (images + 1) / 2
    adjusted = ((image_zero_one - 0.5) * contrast + 0.5).clamp(0, 1)
    adjusted = adjusted * 2 - 1
    return adjusted.squeeze(0) if squeeze_batch else adjusted


def apply_common_degradations(images, config=CommonDegradationConfig()):
    """Apply configured blur, gamma, noise, and contrast in a fixed order."""
    images, squeeze_batch = _as_batch(images)
    _validate_images(images)
    output = images
    if config.blur_sigma:
        output = apply_gaussian_blur(output, config.blur_sigma)
    if config.gamma != 1:
        output = apply_gamma(output, config.gamma)
    if config.noise_std:
        output = apply_gaussian_noise(output, config.noise_std, config.seed)
    if config.contrast != 1:
        output = apply_contrast(output, config.contrast)
    return output.squeeze(0) if squeeze_batch else output


@dataclass(frozen=True)
class RandomFieldASMConfig:
    """Parameters for an ASM degradation with nonuniform random fields."""

    seed: int = 1407
    field_size: int = 8
    d_min: float = 0.3
    d_max: float = 1.0
    beta_min: float = 0.05
    beta_max: float = 0.6
    beta_scale: float = 0.5
    coupling: float = 0.0
    atmospheric_light: tuple[float, float, float] = (0.9, 0.93, 0.96)
    interpolation: Literal['bilinear', 'bicubic'] = 'bicubic'

    def __post_init__(self):
        if self.field_size < 2:
            raise ValueError('field_size must be at least 2')
        if not 0 <= self.d_min <= self.d_max:
            raise ValueError('d_min and d_max must satisfy 0 <= d_min <= d_max')
        if not 0 <= self.beta_min <= self.beta_max:
            raise ValueError(
                'beta_min and beta_max must satisfy 0 <= beta_min <= beta_max')
        if self.beta_scale < 0:
            raise ValueError('beta_scale must be non-negative')
        if not 0 <= self.coupling <= 1:
            raise ValueError('coupling must be in [0, 1]')
        if len(self.atmospheric_light) != 3:
            raise ValueError('atmospheric_light must contain three RGB values')
        if not all(0 <= value <= 1 for value in self.atmospheric_light):
            raise ValueError('atmospheric_light values must be in [0, 1]')

    def to_dict(self):
        return asdict(self)


def _normalize_field(field):
    field_min = field.amin(dim=(-2, -1), keepdim=True)
    field_max = field.amax(dim=(-2, -1), keepdim=True)
    return (field - field_min) / (field_max - field_min).clamp_min(1e-6)


def _random_field(batch_size, height, width, config, device, dtype):
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    low_resolution_shape = (batch_size, 1, config.field_size, config.field_size)
    shared_noise = torch.rand(
        low_resolution_shape, generator=generator, device=device, dtype=dtype)
    depth_noise = torch.rand(
        low_resolution_shape, generator=generator, device=device, dtype=dtype)
    beta_noise = torch.rand(
        low_resolution_shape, generator=generator, device=device, dtype=dtype)
    beta_noise = config.coupling * shared_noise + (1 - config.coupling) * beta_noise

    def resize_and_normalize(noise):
        field = F.interpolate(
            noise, size=(height, width), mode=config.interpolation,
            align_corners=False)
        return _normalize_field(field)

    return resize_and_normalize(depth_noise), resize_and_normalize(beta_noise)


def apply_random_field_asm(images, config=RandomFieldASMConfig()):
    """Apply nonuniform ASM fog to RGB tensors normalized to the [-1, 1] range.

    The same configuration creates identical random fields for equal batch shape,
    device, and dtype. ``beta_scale=0`` returns the input values unchanged.
    """
    if images.ndim == 3:
        images = images.unsqueeze(0)
        squeeze_batch = True
    elif images.ndim == 4:
        squeeze_batch = False
    else:
        raise ValueError('images must have shape [C, H, W] or [B, C, H, W]')
    if images.shape[1] != 3:
        raise ValueError('images must contain exactly three RGB channels')
    if not images.is_floating_point():
        raise TypeError('images must use a floating-point dtype')
    if images.amin() < -1.0001 or images.amax() > 1.0001:
        raise ValueError('images must be normalized to the [-1, 1] range')

    batch_size, _, height, width = images.shape
    if config.beta_scale == 0:
        zero_field = torch.zeros(
            (batch_size, 1, height, width), device=images.device, dtype=images.dtype)
        one_field = torch.ones_like(zero_field)
        result = {
            'image': images.clone().squeeze(0) if squeeze_batch else images.clone(),
            'depth': zero_field.squeeze(0) if squeeze_batch else zero_field,
            'beta': zero_field.squeeze(0) if squeeze_batch else zero_field,
            'transmission': one_field.squeeze(0) if squeeze_batch else one_field,
            'config': config.to_dict(),
        }
        return result

    depth_unit, beta_unit = _random_field(
        batch_size, height, width, config, images.device, images.dtype)
    depth = config.d_min + (config.d_max - config.d_min) * depth_unit
    beta = config.beta_scale * (
        config.beta_min + (config.beta_max - config.beta_min) * beta_unit)
    transmission = torch.exp(-beta * depth)

    image_zero_one = (images + 1) / 2
    atmospheric_light = torch.tensor(
        config.atmospheric_light, device=images.device, dtype=images.dtype
    ).view(1, 3, 1, 1)
    degraded_zero_one = image_zero_one * transmission + atmospheric_light * (1 - transmission)
    degraded = degraded_zero_one.clamp(0, 1) * 2 - 1

    result = {
        'image': degraded.squeeze(0) if squeeze_batch else degraded,
        'depth': depth.squeeze(0) if squeeze_batch else depth,
        'beta': beta.squeeze(0) if squeeze_batch else beta,
        'transmission': transmission.squeeze(0) if squeeze_batch else transmission,
        'config': config.to_dict(),
    }
    return result