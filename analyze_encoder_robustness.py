"""Compare raw Encoder stability under reproducible image degradations."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from model import CUTGenerator
from utils.degradation import (
    CommonDegradationConfig,
    RandomFieldASMConfig,
    apply_common_degradations,
    apply_random_field_asm,
)


MODEL_NAMES = ('cut_baseline', 'expander_decorr_200_exp1_s1407')
ASM_SCALES = {
    'asm_weak': 0.25,
    'asm_medium': 0.50,
    'asm_strong': 0.75,
    'asm_very_strong': 1.00,
}
CONDITIONS = (
    'clean', 'asm_weak', 'asm_medium', 'asm_strong', 'asm_very_strong',
    'blur_05', 'gamma_bright', 'noise', 'contrast_low',
    'asm_medium_blur_gamma',
)


def load_image(path):
    with Image.open(path) as image:
        array = np.asarray(image.convert('RGB'), dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0


def pad_to_multiple(image, multiple=4):
    _, _, height, width = image.shape
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if not pad_height and not pad_width:
        return image
    return F.pad(image, (0, pad_width, 0, pad_height), mode='reflect')


def feature_names(encoder_blocks):
    return [
        'stem', 'downsample_1', 'downsample_2', 'enc_resblock_1',
        f'enc_resblock_{encoder_blocks}',
    ]


def load_model(experiment_name, args, device):
    checkpoint_path = Path('results/ckpt') / experiment_name / 'best_test_psnr.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    generator = CUTGenerator(
        ngf=args.ngf,
        encoder_blocks=args.encoder_blocks,
        decoder_blocks=args.decoder_blocks,
        downsample_mode=args.downsample_mode,
        upsample_mode=args.upsample_mode,
    ).to(device)
    generator.load_state_dict(checkpoint['generator'])
    generator.eval()
    return generator, checkpoint_path, checkpoint.get('epoch')


def make_condition(image, condition, field_seed):
    if condition == 'clean':
        return image.clone()
    if condition in ASM_SCALES:
        asm = apply_random_field_asm(
            image,
            RandomFieldASMConfig(seed=field_seed, beta_scale=ASM_SCALES[condition]),
        )
        return asm['image']
    if condition == 'blur_05':
        return apply_common_degradations(
            image, CommonDegradationConfig(blur_sigma=0.5))
    if condition == 'gamma_bright':
        return apply_common_degradations(
            image, CommonDegradationConfig(gamma=0.9))
    if condition == 'noise':
        return apply_common_degradations(
            image, CommonDegradationConfig(noise_std=0.025, seed=field_seed))
    if condition == 'contrast_low':
        return apply_common_degradations(
            image, CommonDegradationConfig(contrast=0.9))
    if condition == 'asm_medium_blur_gamma':
        asm = apply_random_field_asm(
            image,
            RandomFieldASMConfig(seed=field_seed, beta_scale=0.5),
        )
        return apply_common_degradations(
            asm['image'],
            CommonDegradationConfig(blur_sigma=0.5, gamma=0.9),
        )
    raise ValueError(f'Unknown condition: {condition}')


def pair_metrics(reference, perturbed):
    reference = reference.flatten(2).transpose(1, 2)
    perturbed = perturbed.flatten(2).transpose(1, 2)
    reference_norm = reference.norm(dim=-1)
    perturbed_norm = perturbed.norm(dim=-1)
    cosine = 1 - F.cosine_similarity(reference, perturbed, dim=-1, eps=1e-8)
    normalized_l2 = (reference - perturbed).norm(dim=-1) / (
        0.5 * (reference_norm + perturbed_norm) + 1e-8)
    norm_ratio = perturbed_norm / (reference_norm + 1e-8)
    return {
        'cosine_distance': cosine.mean().item(),
        'normalized_l2': normalized_l2.mean().item(),
        'feature_norm_ratio': norm_ratio.mean().item(),
    }


def summarize(rows):
    values = defaultdict(list)
    for row in rows:
        key = (row['model'], row['condition'], row['layer'])
        values[key].append(row)
    summary = {}
    for (model, condition, layer), entries in values.items():
        summary.setdefault(model, {}).setdefault(condition, {})[layer] = {}
        for metric in ('cosine_distance', 'normalized_l2', 'feature_norm_ratio'):
            numbers = np.asarray([entry[metric] for entry in entries], dtype=np.float64)
            summary[model][condition][layer][metric] = {
                'mean': float(numbers.mean()),
                'std': float(numbers.std(ddof=1)) if len(numbers) > 1 else 0.0,
                'p05': float(np.percentile(numbers, 5)),
                'p50': float(np.percentile(numbers, 50)),
                'p95': float(np.percentile(numbers, 95)),
            }
    return summary


def write_summary_markdown(path, summary, metadata):
    lines = [
        '# Raw Encoder Robustness Summary', '',
        f'- Images: {metadata["images"]}',
        f'- Conditions: {", ".join(metadata["conditions"])}',
        f'- Layers: {", ".join(metadata["layers"])}', '',
        'The primary metric is same-location patch cosine distance. '
        'Normalized L2 and feature norm ratio are reported as safeguards.', '',
        '| Model | Condition | Layer | Cosine mean | Cosine std | NormL2 mean | Norm ratio mean |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for model in metadata['models']:
        for condition in metadata['conditions']:
            for layer in metadata['layers']:
                metrics = summary[model][condition][layer]
                lines.append(
                    f'| `{model}` | `{condition}` | `{layer}` | '
                    f'{metrics["cosine_distance"]["mean"]:.6f} | '
                    f'{metrics["cosine_distance"]["std"]:.6f} | '
                    f'{metrics["normalized_l2"]["mean"]:.6f} | '
                    f'{metrics["feature_norm_ratio"]["mean"]:.6f} |')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--device', default=config.DEVICE)
    parser.add_argument('--ngf', type=int, default=config.NGF)
    parser.add_argument('--encoder-blocks', type=int, default=config.ENCODER_BLOCKS)
    parser.add_argument('--decoder-blocks', type=int, default=config.DECODER_BLOCKS)
    parser.add_argument('--downsample-mode', default=config.DOWN_SAMPLE_MODE)
    parser.add_argument('--upsample-mode', default=config.UP_SAMPLE_MODE)
    parser.add_argument('--data-root', default='datasets/RRSHID/testA')
    parser.add_argument('--output-dir', default='results/feature_analysis/encoder_robustness')
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error('--max-images must be non-negative')
    return args


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu')
    paths = sorted(Path(args.data_root).glob('*.png'))
    if args.max_images:
        paths = paths[:args.max_images]
    if not paths:
        raise FileNotFoundError(f'No PNG images found in {args.data_root}')

    models = {}
    checkpoints = {}
    epochs = {}
    for model_name in MODEL_NAMES:
        models[model_name], checkpoints[model_name], epochs[model_name] = load_model(
            model_name, args, device)
    layers = feature_names(args.encoder_blocks)
    rows = []
    for image_index, path in enumerate(paths):
        image = pad_to_multiple(load_image(path).unsqueeze(0).to(device))
        field_seed = 1407 + image_index
        clean_features = {}
        for model_name, generator in models.items():
            _, clean_features[model_name] = generator.encode(
                image, return_features=True)
        for condition in CONDITIONS:
            perturbed = make_condition(image, condition, field_seed)
            for model_name, generator in models.items():
                _, perturbed_features = generator.encode(
                    perturbed, return_features=True)
                for layer in layers:
                    metrics = pair_metrics(
                        clean_features[model_name][layer], perturbed_features[layer])
                    rows.append({
                        'model': model_name,
                        'image': path.name,
                        'condition': condition,
                        'field_seed': field_seed,
                        'layer': layer,
                        **metrics,
                    })
        if (image_index + 1) % 10 == 0 or image_index + 1 == len(paths):
            print(f'processed {image_index + 1}/{len(paths)} images')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'images': len(paths),
        'data_root': args.data_root,
        'models': list(MODEL_NAMES),
        'checkpoints': {name: str(path) for name, path in checkpoints.items()},
        'epochs': epochs,
        'layers': layers,
        'conditions': list(CONDITIONS),
        'field_seed_rule': '1407 + sorted image index',
        'preprocessing': {'normalization': 'pixel / 127.5 - 1.0', 'padding': 'reflect to multiple of 4'},
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    with (output_dir / 'per_image_metrics.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    summary = summarize(rows)
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    write_summary_markdown(output_dir / 'summary.md', summary, metadata)
    print(f'Analyzed {len(paths)} images')
    print(f'Results: {output_dir}')


if __name__ == '__main__':
    main()
