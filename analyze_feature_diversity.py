"""Analyze diversity and richness of the saved CUT NCE feature space."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from analyze_features import get_nce_feature_names, load_image, pad_to_multiple
from model import CUTGenerator, PatchSampleF


def get_patch_ids(features, feature_names, max_patches):
    patch_ids = []
    for name in feature_names:
        locations = features[name].shape[2] * features[name].shape[3]
        count = min(locations, max_patches)
        patch_ids.append(torch.linspace(
            0, locations - 1, steps=count, device=features[name].device
        ).long())
    return patch_ids


def load_model(args, device):
    checkpoint_path = Path('results/ckpt') / args.experiment_name / 'best_test_psnr.pt'
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
    feature_names = get_nce_feature_names(args.encoder_blocks)
    with torch.inference_mode():
        _, initial_features = generator(
            torch.zeros(1, 3, args.crop_size, args.crop_size, device=device),
            return_features=True)
    feature_net = PatchSampleF(nc=256).to(device)
    feature_net.initialize([initial_features[name] for name in feature_names])
    feature_net.load_state_dict(checkpoint['feature_net'])
    feature_net.eval()
    return generator, feature_net, feature_names, checkpoint_path, checkpoint.get('epoch')


def append_features(store, feature_net, features, feature_names, patch_ids):
    _, _, projected = feature_net(
        [features[name] for name in feature_names],
        num_patches=1,
        patch_ids=patch_ids,
        return_projected=True,
    )
    for name, values in zip(feature_names, projected):
        store[name].append(values.detach().cpu().float())


def covariance_stats(values):
    values = values.float()
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum().clamp_min(1e-12)
    probabilities = eigenvalues / total
    effective_rank = torch.exp(-(
        probabilities * torch.log(probabilities.clamp_min(1e-12))).sum())
    participation_ratio = total.square() / eigenvalues.square().sum().clamp_min(1e-12)
    pc1_ratio = (eigenvalues[-1] / total).item()
    std = covariance.diagonal().clamp_min(0.0).sqrt()
    correlation = covariance / (std[:, None] * std[None, :] + 1e-6)
    off_diagonal = correlation - torch.diag_embed(torch.diagonal(correlation))
    mean_abs_corr = off_diagonal.abs().sum() / max(values.shape[1] * (values.shape[1] - 1), 1)
    active_dimensions = (eigenvalues > total * 1e-3 / values.shape[1]).sum().item()
    return {
        'samples': values.shape[0],
        'mean_std': std.mean().item(),
        'min_std': std.min().item(),
        'effective_rank_entropy': effective_rank.item(),
        'participation_ratio': participation_ratio.item(),
        'pc1_ratio': pc1_ratio,
        'active_dimensions_0.1_percent': active_dimensions,
        'mean_abs_channel_correlation': mean_abs_corr.item(),
    }


def cosine_diversity(values, max_pairs, seed):
    values = F.normalize(values.float(), dim=1)
    generator = torch.Generator().manual_seed(seed)
    pair_count = min(max_pairs, values.shape[0] * (values.shape[0] - 1) // 2)
    if pair_count == 0:
        return {'mean_cosine_distance': 0.0, 'std_cosine_distance': 0.0,
                'p05_cosine_distance': 0.0, 'p50_cosine_distance': 0.0,
                'p95_cosine_distance': 0.0}
    left = torch.randint(values.shape[0], (pair_count,), generator=generator)
    right = torch.randint(values.shape[0], (pair_count,), generator=generator)
    different = left != right
    left, right = left[different], right[different]
    distances = 1.0 - (values[left] * values[right]).sum(dim=1)
    quantiles = torch.quantile(distances, torch.tensor([0.05, 0.50, 0.95]))
    return {
        'mean_cosine_distance': distances.mean().item(),
        'std_cosine_distance': distances.std(unbiased=False).item(),
        'p05_cosine_distance': quantiles[0].item(),
        'p50_cosine_distance': quantiles[1].item(),
        'p95_cosine_distance': quantiles[2].item(),
    }


def summarize(stores, max_pairs, seed):
    summary = {}
    for group, representation_stores in stores.items():
        summary[group] = {}
        for representation, layer_store in representation_stores.items():
            summary[group][representation] = {}
            for layer, chunks in layer_store.items():
                values = torch.cat(chunks, dim=0)
                summary[group][representation][layer] = {
                    **covariance_stats(values),
                    **cosine_diversity(values, max_pairs, seed),
                }
    return summary


@torch.inference_mode()
def analyze(args, device):
    generator, feature_net, feature_names, checkpoint_path, epoch = load_model(args, device)
    source_paths = {path.name: path for path in Path(args.data_root, 'testA').glob('*.png')}
    target_paths = {path.name: path for path in Path(args.data_root, 'testB').glob('*.png')}
    names = sorted(source_paths.keys() & target_paths.keys())
    if args.max_images:
        names = names[:args.max_images]
    stores = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for name in names:
        source = load_image(source_paths[name]).unsqueeze(0).to(device)
        target = load_image(target_paths[name]).unsqueeze(0).to(device)
        if source.shape != target.shape:
            continue
        source, height, width = pad_to_multiple(source)
        target, _, _ = pad_to_multiple(target)
        source_bottleneck, source_features = generator.encode(source, return_features=True)
        target_bottleneck, target_features = generator.encode(target, return_features=True)
        generated = generator.decode(source_bottleneck)
        _, generated_features = generator.encode(generated, return_features=True)
        patch_ids = get_patch_ids(source_features, feature_names, args.max_patches)
        domain = name.split('_', 1)[0]
        for group, features in (
            ('A_fog', source_features),
            ('B_clear', target_features),
            ('G_A', generated_features),
        ):
            append_features(
                stores['all'][group], feature_net, features, feature_names, patch_ids)
            append_features(
                stores[domain][group], feature_net, features, feature_names, patch_ids)
    return {
        'experiment': args.experiment_name,
        'checkpoint': str(checkpoint_path),
        'epoch': epoch,
        'images': len(names),
        'groups': sorted(stores.keys()),
        'feature_names': feature_names,
        'metrics': summarize(stores, args.max_pairs, args.seed),
    }


def write_report(path, result):
    lines = [
        '# NCE Feature Diversity and Richness',
        '',
        f'- Experiment: `{result["experiment"]}`',
        f'- Checkpoint: `{result["checkpoint"]}`',
        f'- Epoch: {result["epoch"]}',
        f'- Paired images analyzed: {result["images"]}',
        '',
        'Effective rank and participation ratio measure feature richness after centering.',
        'Cosine distance statistics measure sample/patch diversity in the normalized NCE space.',
        'Higher effective rank, participation ratio, and cosine distance indicate richer or more diverse representations.',
        '',
    ]
    for group in result['groups']:
        lines.extend([f'## {group}', ''])
        for representation, layer_metrics in result['metrics'][group].items():
            lines.extend([
                f'### {representation}', '',
                '| Layer | Samples | Effective rank | Participation ratio | PC1 ratio | Mean abs corr | Mean cosine distance | P05 | P50 | P95 |',
                '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
            ])
            for layer, metrics in layer_metrics.items():
                lines.append(
                    f'| `{layer}` | {metrics["samples"]} | {metrics["effective_rank_entropy"]:.2f} | '
                    f'{metrics["participation_ratio"]:.2f} | {metrics["pc1_ratio"]:.4f} | '
                    f'{metrics["mean_abs_channel_correlation"]:.4f} | {metrics["mean_cosine_distance"]:.4f} | '
                    f'{metrics["p05_cosine_distance"]:.4f} | {metrics["p50_cosine_distance"]:.4f} | '
                    f'{metrics["p95_cosine_distance"]:.4f} |')
            lines.append('')
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze CUT NCE feature diversity')
    parser.add_argument('experiment_name')
    parser.add_argument('--data-root', default=config.DATA_ROOT)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--max-patches', type=int, default=256)
    parser.add_argument('--max-pairs', type=int, default=100000)
    parser.add_argument('--device', default=config.DEVICE)
    parser.add_argument('--ngf', type=int, default=config.NGF)
    parser.add_argument('--encoder-blocks', type=int, default=config.ENCODER_BLOCKS)
    parser.add_argument('--decoder-blocks', type=int, default=config.DECODER_BLOCKS)
    parser.add_argument('--downsample-mode', default=config.DOWN_SAMPLE_MODE)
    parser.add_argument('--upsample-mode', default=config.UP_SAMPLE_MODE)
    parser.add_argument('--crop-size', type=int, default=config.CROP_SIZE)
    parser.add_argument('--seed', type=int, default=1407)
    args = parser.parse_args()
    if args.max_images < 0 or args.max_patches < 1 or args.max_pairs < 1:
        parser.error('max-images must be non-negative; max-patches and max-pairs must be positive')
    return args


def main():
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run with --device cpu')
    device = torch.device(args.device)
    result = analyze(args, device)
    output_dir = Path('results/feature_analysis') / args.experiment_name / Path(args.data_root).name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'feature_diversity.json'
    output_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    report_path = output_dir / 'feature_diversity.md'
    write_report(report_path, result)
    print(f'Analyzed {result["images"]} paired images')
    print(f'JSON: {output_path}')
    print(f'Report: {report_path}')


if __name__ == '__main__':
    main()