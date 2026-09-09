from pathlib import Path
import math
import random
import warnings

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset


class UnpairedImageDataset(Dataset):
    """Independent A/B image sampling for CUT training."""

    def __init__(self, root, phase='train', crop_size=256, load_size=None,
                 augment=None, weak_degradation=True):
        root = Path(root)
        self.paths_a = self._valid_paths(root / f'{phase}A')
        self.paths_b = self._valid_paths(root / f'{phase}B')
        if not self.paths_a or not self.paths_b:
            raise FileNotFoundError(
                f'Need valid images in both {phase}A and {phase}B; '
                f'found {len(self.paths_a)} and {len(self.paths_b)} in {root}'
            )
        if crop_size % 4 != 0:
            raise ValueError('crop_size must be divisible by 4')
        self.crop_size = crop_size
        self.load_size = load_size or math.ceil(crop_size * 1.12)
        if self.load_size < crop_size:
            raise ValueError('load_size must be greater than or equal to crop_size')
        self.augment = phase == 'train' if augment is None else augment
        self.weak_degradation = weak_degradation and self.augment

    @staticmethod
    def _valid_paths(directory):
        paths = sorted(directory.glob('*.png'))
        valid_paths = []
        invalid_count = 0
        for path in paths:
            try:
                with Image.open(path) as image:
                    image.verify()
                valid_paths.append(path)
            except (OSError, ValueError):
                invalid_count += 1
        if invalid_count:
            warnings.warn(
                f'Skipped {invalid_count} invalid images in {directory}',
                RuntimeWarning,
            )
        return valid_paths

    def __len__(self):
        return max(len(self.paths_a), len(self.paths_b))

    def _resize_and_crop(self, image):
        scale = max(self.load_size / image.width, self.load_size / image.height)
        resized_size = (math.ceil(image.width * scale), math.ceil(image.height * scale))
        image = image.resize(resized_size, Image.Resampling.BICUBIC)
        if self.augment:
            left = random.randint(0, image.width - self.crop_size)
            top = random.randint(0, image.height - self.crop_size)
        else:
            left = (image.width - self.crop_size) // 2
            top = (image.height - self.crop_size) // 2
        return image.crop((left, top, left + self.crop_size, top + self.crop_size))

    @staticmethod
    def _apply_weak_degradation(image):
        operation = random.choice(('blur', 'noise', 'resample'))
        if operation == 'blur':
            return image.filter(ImageFilter.GaussianBlur(random.uniform(0.1, 0.35)))
        if operation == 'resample':
            scale = random.uniform(0.85, 0.95)
            small_size = (max(1, round(image.width * scale)),
                          max(1, round(image.height * scale)))
            return image.resize(small_size, Image.Resampling.BICUBIC).resize(
                image.size, Image.Resampling.BICUBIC)
        pixels = np.asarray(image, dtype=np.float32)
        pixels += np.random.normal(0.0, random.uniform(0.5, 1.5), pixels.shape)
        return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))

    def _load(self, path, apply_degradation=False):
        image = Image.open(path).convert('RGB')
        image = self._resize_and_crop(image)
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if apply_degradation and random.random() < 0.3:
            image = self._apply_weak_degradation(image)
        tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0
        return tensor

    def __getitem__(self, index):
        path_a = self.paths_a[index % len(self.paths_a)]
        path_b = self.paths_b[random.randrange(len(self.paths_b))]
        return {'A': self._load(path_a, self.weak_degradation), 'B': self._load(path_b),
                'A_paths': str(path_a), 'B_paths': str(path_b)}