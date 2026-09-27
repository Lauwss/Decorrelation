"""Compare A/B/G(A) domain geometry in the raw Encoder feature space."""

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


MODEL_NAMES = ('cut_baseline', 'expander_decorr_200_exp1_s1407')


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


def pair_metrics(first, second):
    first = first.flatten(2).transpose(1, 2)
    second = second.flatten(2).transpose(1, 2)
    first_norm = first.norm(dim=-1)
    second_norm = second.norm(dim=-1)
    cosine_distance = 1 - F.cosine_similarity(
        first, second, dim=-1, eps=1e-8)
    normalized_l2 = (first - second).norm(dim=-1) / (
        0.5 * (first_norm + second_norm) + 1e-8)
    norm_ratio = second_norm / (first_norm + 1e-8)
    return {
        'cosine_distance': cosine_distance.mean().item(),
        'normalized_l2': normalized_l2.mean().item(),
        'feature_norm_ratio': norm_ratio.mean().item(),
    }


def summarize(rows, layers):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['model'], row['layer'])].append(row)
    summary = {}
    for (model, layer), entries in grouped.items():
        summary.setdefault(model, {})[layer] = {}
        for metric in (
            'ab_cosine_distance', 'ag_cosine_distance', 'bg_cosine_distance',
            'source_preference', 'ab_normalized_l2', 'ag_normalized_l2',
            'bg_normalized_l2', 'ag_feature_norm_ratio', 'bg_feature_norm_ratio',
        ):
            values = np.asarray([entry[metric] for entry in entries], dtype=np.float64)
            summary[model][layer][metric] = {
                'mean': float(values.mean()),
                'std': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                'p05': float(np.percentile(values, 5)),
                'p50': float(np.percentile(values, 50)),
                'p95': float(np.percentile(values, 95)),
            }
    for model in MODEL_NAMES:
        for layer in layers:
            if model not in summary or layer not in summary[model]:
                raise RuntimeError(f'Missing summary for {model}/{layer}')
    return summary


def write_summary(path, summary, metadata):
    lines = [
        '# Raw Encoder Domain Alignment', '',
        f'- Images: {metadata["images"]}',
        f'- Layers: {", ".join(metadata["layers"])}',
        '- No image degradation was applied.', '',
        'For each layer, all metrics compare features at the same spatial locations.',
        'The source preference is $d(B,G(A))-d(A,G(A))$; positive values mean',
        '$G(A)$ is closer to the foggy source $A$.', '',
        '| Model | Layer | $d(A,B)$ | $d(A,G(A))$ | $d(B,G(A))$ | Source preference |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for model in metadata['models']:
        for layer in metadata['layers']:
            values = summary[model][layer]
            lines.append(
                f'| `{model}` | `{layer}` | '
                f'{values["ab_cosine_distance"]["mean"]:.6f} | '
                f'{values["ag_cosine_distance"]["mean"]:.6f} | '
                f'{values["bg_cosine_distance"]["mean"]:.6f} | '
                f'{values["source_preference"]["mean"]:.6f} |')
    lines.extend([
        '',
        'The primary metric is cosine distance $1 - \hat f_1^T\hat f_2$. '
        'Normalized L2 and feature norm ratios are included in `summary.json` '
        'as checks against scale-only effects.', '',
        '## Metric Interpretation', '',
        '- $d(A,B)$: raw Encoder separation of the paired domains.',
        '- $d(A,G(A))$: distance from the generated image to its source input.',
        '- $d(B,G(A))$: distance from the generated image to the paired clear image.',
        '- Positive source preference: the generated image remains closer to $A$ than $B$.',
    ])
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
    parser.add_argument('--data-root', default=config.DATA_ROOT)
    parser.add_argument('--output-dir', default='results/feature_analysis/encoder_domain_alignment')
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error('--max-images must be non-negative')
    return args


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --device cpu')
    source_paths = {path.name: path for path in (Path(args.data_root) / 'testA').glob('*.png')}
    target_paths = {path.name: path for path in (Path(args.data_root) / 'testB').glob('*.png')}
    names = sorted(source_paths.keys() & target_paths.keys())
    if args.max_images:
        names = names[:args.max_images]
    if not names:
        raise FileNotFoundError('No same-named PNG pairs found')

    models = {}
    checkpoints = {}
    epochs = {}
    for model_name in MODEL_NAMES:
        models[model_name], checkpoints[model_name], epochs[model_name] = load_model(
            model_name, args, device)
    layers = feature_names(args.encoder_blocks)
    rows = []
    for image_index, name in enumerate(names):
        source = pad_to_multiple(load_image(source_paths[name]).unsqueeze(0).to(device))
        target = pad_to_multiple(load_image(target_paths[name]).unsqueeze(0).to(device))
        if source.shape != target.shape:
            continue
        for model_name, generator in models.items():
            source_bottleneck, source_features = generator.encode(
                source, return_features=True)
            _, target_features = generator.encode(target, return_features=True)
            generated = generator.decode(source_bottleneck)
            _, generated_features = generator.encode(generated, return_features=True)
            for layer in layers:
                ab = pair_metrics(source_features[layer], target_features[layer])
                ag = pair_metrics(source_features[layer], generated_features[layer])
                bg = pair_metrics(target_features[layer], generated_features[layer])
                rows.append({
                    'model': model_name,
                    'image': name,
                    'layer': layer,
                    'image_index': image_index,
                    'ab_cosine_distance': ab['cosine_distance'],
                    'ag_cosine_distance': ag['cosine_distance'],
                    'bg_cosine_distance': bg['cosine_distance'],
                    'source_preference': bg['cosine_distance'] - ag['cosine_distance'],
                    'ab_normalized_l2': ab['normalized_l2'],
                    'ag_normalized_l2': ag['normalized_l2'],
                    'bg_normalized_l2': bg['normalized_l2'],
                    'ag_feature_norm_ratio': ag['feature_norm_ratio'],
                    'bg_feature_norm_ratio': bg['feature_norm_ratio'],
                })
        if (image_index + 1) % 10 == 0 or image_index + 1 == len(names):
            print(f'processed {image_index + 1}/{len(names)} pairs')

    if not rows:
        raise RuntimeError('No valid paired images were analyzed')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'images': len(names),
        'data_root': args.data_root,
        'models': list(MODEL_NAMES),
        'checkpoints': {name: str(path) for name, path in checkpoints.items()},
        'epochs': epochs,
        'layers': layers,
        'preprocessing': {
            'normalization': 'pixel / 127.5 - 1.0',
            'padding': 'reflect to multiple of 4',
            'degradation': 'none',
        },
        'rows': len(rows),
    }
    (output_dir / 'manifest.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    with (output_dir / 'per_image_metrics.jsonl').open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    summary = summarize(rows, layers)
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    write_summary(output_dir / 'summary.md', summary, metadata)
    print(f'Analyzed {len(names)} paired images')
    print(f'Results: {output_dir}')


if __name__ == '__main__':
    main()