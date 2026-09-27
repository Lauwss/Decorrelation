"""Diagnose fog retention at CUT encoder layers and decoder reconstruction."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config
from model import CUTGenerator, PatchSampleF
from utils.PSNR_evaluate import calculate_psnr


def load_image(path):
    image = Image.open(path).convert('RGB')
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
    return tensor / 127.5 - 1.0


def pad_to_multiple(tensor, multiple=4):
    _, _, height, width = tensor.shape
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if not pad_height and not pad_width:
        return tensor, height, width
    return F.pad(tensor, (0, pad_width, 0, pad_height), mode='reflect'), height, width


def normalized_feature_distance(fog_features, clear_features):
    distances = {}
    for name, fog_feature in fog_features.items():
        clear_feature = clear_features[name]
        difference_energy = (fog_feature - clear_feature).square().mean()
        feature_energy = 0.5 * (
            fog_feature.square().mean() + clear_feature.square().mean())
        distances[name] = (difference_energy / feature_energy.clamp_min(1e-12)).sqrt().item()
    return distances


def get_nce_feature_names(encoder_blocks):
    return ['stem', 'downsample_1', 'downsample_2', 'enc_resblock_1',
            f'enc_resblock_{encoder_blocks}']


def get_patch_ids(features, feature_names, max_patches=256):
    patch_ids = []
    for name in feature_names:
        feature = features[name]
        locations = feature.shape[2] * feature.shape[3]
        patch_count = min(locations, max_patches)
        patch_ids.append(torch.linspace(
            0, locations - 1, steps=patch_count, device=feature.device).long())
    return patch_ids


def nce_embeddings(feature_net, features, feature_names, patch_ids):
    embeddings, _ = feature_net(
        [features[name] for name in feature_names], num_patches=1,
        patch_ids=patch_ids)
    return embeddings


def embedding_distance(first_embeddings, second_embeddings, feature_names):
    return {
        name: (1 - (first * second).sum(dim=1)).mean().item()
        for name, first, second in zip(
            feature_names, first_embeddings, second_embeddings)
    }


def load_generator(args, device):
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
    if 'feature_net' not in checkpoint:
        raise KeyError(
            f'Checkpoint does not contain feature_net: {checkpoint_path}')
    feature_names = get_nce_feature_names(args.encoder_blocks)
    with torch.inference_mode():
        _, initialization_features = generator(
            torch.zeros(1, 3, args.crop_size, args.crop_size, device=device),
            return_features=True)
    feature_net = PatchSampleF(nc=256).to(device)
    feature_net.initialize(
        [initialization_features[name] for name in feature_names])
    feature_net.load_state_dict(checkpoint['feature_net'])
    feature_net.eval()
    return generator, feature_net, feature_names, checkpoint_path, checkpoint.get('epoch')


@torch.inference_mode()
def analyze(generator, feature_net, feature_names, source_dir, target_dir, device,
            max_images):
    source_paths = {path.name: path for path in Path(source_dir).glob('*.png')}
    target_paths = {path.name: path for path in Path(target_dir).glob('*.png')}
    names = sorted(source_paths.keys() & target_paths.keys())
    if max_images:
        names = names[:max_images]
    if not names:
        raise FileNotFoundError('No same-named PNG pairs found for feature analysis')

    layer_sums = {}
    nce_layer_sums = {}
    source_generated_nce_sums = {}
    clear_generated_nce_sums = {}
    source_preference_sums = {}
    translation_scores = []
    identity_scores = []
    rows = []
    for name in names:
        fog = load_image(source_paths[name]).unsqueeze(0).to(device)
        clear = load_image(target_paths[name]).unsqueeze(0).to(device)
        if fog.shape != clear.shape:
            continue
        fog, height, width = pad_to_multiple(fog)
        clear, _, _ = pad_to_multiple(clear)
        fog_bottleneck, fog_features = generator.encode(fog, return_features=True)
        clear_bottleneck, clear_features = generator.encode(clear, return_features=True)
        translated = generator.decode(fog_bottleneck)[:, :, :height, :width]
        identity = generator.decode(clear_bottleneck)[:, :, :height, :width]
        target = clear[:, :, :height, :width]
        _, generated_features = generator.encode(
            generator.decode(fog_bottleneck), return_features=True)

        distances = normalized_feature_distance(fog_features, clear_features)
        patch_ids = get_patch_ids(fog_features, feature_names)
        fog_embeddings = nce_embeddings(
            feature_net, fog_features, feature_names, patch_ids)
        clear_embeddings = nce_embeddings(
            feature_net, clear_features, feature_names, patch_ids)
        generated_embeddings = nce_embeddings(
            feature_net, generated_features, feature_names, patch_ids)
        nce_distances = embedding_distance(
            fog_embeddings, clear_embeddings, feature_names)
        source_generated_distances = embedding_distance(
            fog_embeddings, generated_embeddings, feature_names)
        clear_generated_distances = embedding_distance(
            clear_embeddings, generated_embeddings, feature_names)
        source_preference = {
            layer: clear_generated_distances[layer] - source_generated_distances[layer]
            for layer in feature_names
        }
        for layer, value in distances.items():
            layer_sums[layer] = layer_sums.get(layer, 0.0) + value
        for layer, value in nce_distances.items():
            nce_layer_sums[layer] = nce_layer_sums.get(layer, 0.0) + value
        for layer, value in source_generated_distances.items():
            source_generated_nce_sums[layer] = (
                source_generated_nce_sums.get(layer, 0.0) + value)
        for layer, value in clear_generated_distances.items():
            clear_generated_nce_sums[layer] = (
                clear_generated_nce_sums.get(layer, 0.0) + value)
        for layer, value in source_preference.items():
            source_preference_sums[layer] = (
                source_preference_sums.get(layer, 0.0) + value)
        translation_psnr = calculate_psnr(translated, target)
        identity_psnr = calculate_psnr(identity, target)
        translation_scores.append(translation_psnr)
        identity_scores.append(identity_psnr)
        rows.append({
            'image': name,
            'translation_psnr': translation_psnr,
            'identity_psnr': identity_psnr,
            'decoder_oracle_gap': identity_psnr - translation_psnr,
            'feature_distances': distances,
            'nce_embedding_distances': nce_distances,
            'nce_source_to_generated_distance': source_generated_distances,
            'nce_clear_to_generated_distance': clear_generated_distances,
            'nce_source_preference': source_preference,
        })

    count = len(rows)
    if not count:
        raise RuntimeError('No valid same-sized image pairs were analyzed')
    layer_distances = {layer: value / count for layer, value in layer_sums.items()}
    nce_layer_distances = {
        layer: value / count for layer, value in nce_layer_sums.items()}
    source_generated_nce_distances = {
        layer: value / count for layer, value in source_generated_nce_sums.items()}
    clear_generated_nce_distances = {
        layer: value / count for layer, value in clear_generated_nce_sums.items()}
    source_preference = {
        layer: value / count for layer, value in source_preference_sums.items()}
    return {
        'images': count,
        'mean_translation_psnr': sum(translation_scores) / count,
        'mean_identity_psnr': sum(identity_scores) / count,
        'mean_decoder_oracle_gap': (sum(identity_scores) - sum(translation_scores)) / count,
        'mean_feature_distance': layer_distances,
        'mean_nce_embedding_distance': nce_layer_distances,
        'mean_nce_source_to_generated_distance': source_generated_nce_distances,
        'mean_nce_clear_to_generated_distance': clear_generated_nce_distances,
        'mean_nce_source_preference': source_preference,
        'per_image': rows,
    }


def write_report(path, experiment_name, checkpoint_path, epoch, result):
    lines = [
        '# CUT Feature Retention Diagnosis',
        '',
        f'- Experiment: `{experiment_name}`',
        f'- Checkpoint: `{checkpoint_path}`',
        f'- Epoch: {epoch if epoch is not None else "unknown"}',
        f'- Images: {result["images"]}',
        f'- Translation PSNR: {result["mean_translation_psnr"]:.4f} dB',
        f'- Clear-domain identity PSNR: {result["mean_identity_psnr"]:.4f} dB',
        f'- Decoder oracle gap: {result["mean_decoder_oracle_gap"]:.4f} dB',
        '',
        '## Encoder Fog Sensitivity',
        '',
        'The normalized distance is $||E_l(A)-E_l(B)||_2$ divided by the mean feature energy.',
        'A large value in deep layers means the encoder still distinguishes fog from its clear counterpart.',
        '',
        '| Encoder layer | Mean normalized feature distance |',
        '| --- | ---: |',
    ]
    for layer, value in result['mean_feature_distance'].items():
        lines.append(f'| `{layer}` | {value:.4f} |')
    lines.extend([
        '',
        '## PatchNCE Expander Space',
        '',
        'All same-location patches are projected by the saved expander and $L_2$ normalized',
        'exactly as in training. The metric is mean cosine distance',
        '$1 - \\hat z_l(A)^T \\hat z_l(B)$; a large value means the NCE space distinguishes',
        'fog from its clear counterpart.',
        '',
        '| PatchNCE layer | $d(A,B)$ | $d(A,G(A))$ | $d(B,G(A))$ | Source preference |',
        '| --- | ---: | ---: | ---: | ---: |',
    ])
    for layer, value in result['mean_nce_embedding_distance'].items():
        source_value = result['mean_nce_source_to_generated_distance'][layer]
        clear_value = result['mean_nce_clear_to_generated_distance'][layer]
        preference = result['mean_nce_source_preference'][layer]
        lines.append(
            f'| `{layer}` | {value:.4f} | {source_value:.4f} | '
            f'{clear_value:.4f} | {preference:.4f} |')
    lines.extend([
        '',
        'Here $d(A,B)$ compares the paired input domains. The other two columns compare',
        'the generated image with each domain at the same patch locations.',
        'Source preference is $d(B,G(A))-d(A,G(A))$; positive values mean $G(A)$ is',
        'closer to the foggy source than to the paired clear target.',
    ])
    lines.extend([
        '',
        '## Interpretation',
        '',
        'High identity PSNR with persistent deep-layer distance indicates encoder fog retention.',
        'Low identity PSNR indicates decoder clear-domain reconstruction is a limiting factor.',
        'In the saved PatchNCE space, a positive source preference shows the generated image',
        'stays aligned with its foggy source rather than with the paired clear reference.',
        'Large positive preference is expected whenever PatchNCE dominates the representation.',
        '',
        '## Per-image Results',
        '',
        '| Image | Translation PSNR | Identity PSNR | Decoder oracle gap |',
        '| --- | ---: | ---: | ---: |',
    ])
    for row in result['per_image']:
        lines.append(
            f'| `{row["image"]}` | {row["translation_psnr"]:.4f} dB '
            f'| {row["identity_psnr"]:.4f} dB | {row["decoder_oracle_gap"]:.4f} dB |')
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args():
    parser = argparse.ArgumentParser(description='Diagnose CUT encoder fog retention')
    parser.add_argument('experiment_name')
    parser.add_argument('--data-root', default=config.DATA_ROOT)
    parser.add_argument('--max-images', type=int, default=0)
    parser.add_argument('--device', default=config.DEVICE)
    parser.add_argument('--ngf', type=int, default=config.NGF)
    parser.add_argument('--encoder-blocks', type=int, default=config.ENCODER_BLOCKS)
    parser.add_argument('--decoder-blocks', type=int, default=config.DECODER_BLOCKS)
    parser.add_argument('--downsample-mode', default=config.DOWN_SAMPLE_MODE)
    parser.add_argument('--upsample-mode', default=config.UP_SAMPLE_MODE)
    parser.add_argument('--crop-size', type=int, default=config.CROP_SIZE)
    args = parser.parse_args()
    if Path(args.experiment_name).name != args.experiment_name:
        parser.error('experiment_name must be a directory name, not a path')
    if args.max_images < 0:
        parser.error('--max-images must be non-negative')
    return args


def main():
    args = parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; run with --device cpu or use a GPU node')
    generator, feature_net, feature_names, checkpoint_path, epoch = load_generator(
        args, torch.device(args.device))
    result = analyze(
        generator,
        feature_net,
        feature_names,
        Path(args.data_root) / 'testA',
        Path(args.data_root) / 'testB',
        torch.device(args.device),
        args.max_images,
    )
    dataset_name = Path(args.data_root).name
    output_dir = Path('results/feature_analysis') / args.experiment_name / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'metrics.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    write_report(output_dir / 'report.md', args.experiment_name, checkpoint_path, epoch, result)
    print(f'Analyzed {result["images"]} paired images')
    print(f'Translation PSNR: {result["mean_translation_psnr"]:.4f} dB')
    print(f'Identity PSNR: {result["mean_identity_psnr"]:.4f} dB')
    print(f'Report: {output_dir / "report.md"}')


if __name__ == '__main__':
    main()