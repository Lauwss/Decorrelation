import argparse
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import config
from model import CUTGenerator
from utils.PSNR_evaluate import calculate_psnr


def load_image(path):
    image = Image.open(path).convert('RGB')
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
    return tensor / 127.5 - 1.0


def save_tensor_image(tensor, path):
    pixels = ((tensor.detach().cpu().float() + 1.0) * 127.5).clamp(0, 255)
    pixels = pixels.permute(1, 2, 0).numpy().astype(np.uint8)
    Image.fromarray(pixels).save(path)


def pad_to_multiple(tensor, multiple=4):
    _, _, height, width = tensor.shape
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if not pad_height and not pad_width:
        return tensor, height, width
    return F.pad(tensor, (0, pad_width, 0, pad_height), mode='reflect'), height, width


def create_overview(rows, output_path, thumbnail_size=(320, 240)):
    column_width, image_height = thumbnail_size
    header_height = 36
    canvas = Image.new(
        'RGB',
        (column_width * 3, header_height + image_height * len(rows)),
        'white',
    )
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(('Input A', 'Generated B', 'Target B')):
        draw.text((column * column_width + 8, 10), title, fill='black')
    for row_index, row in enumerate(rows):
        for column, key in enumerate(('input', 'output', 'target')):
            if row[key] is None:
                left = column * column_width + column_width // 2 - 12
                top = header_height + row_index * image_height + image_height // 2
                draw.text((left, top), 'N/A', fill='gray')
                continue
            image = Image.open(row[key]).convert('RGB')
            image.thumbnail((column_width, image_height), Image.Resampling.LANCZOS)
            left = column * column_width + (column_width - image.width) // 2
            top = header_height + row_index * image_height
            top += (image_height - image.height) // 2
            canvas.paste(image, (left, top))
    canvas.save(output_path)


def write_markdown_report(output_dir, checkpoint_path, epoch, rows, mean_psnr):
    def relative_link(path):
        return quote(Path(path).relative_to(output_dir).as_posix())

    lines = [
        '# CUT Test Visualization',
        '',
        f'- Checkpoint: `{checkpoint_path}`',
        f'- Epoch: {epoch if epoch is not None else "unknown"}',
        f'- Images: {len(rows)}',
        f'- Mean PSNR: {mean_psnr:.4f} dB' if mean_psnr is not None
        else '- Mean PSNR: unavailable (no same-named target images)',
        '',
        '## Overview',
        '',
        '![Input, generated output, and target comparison](overview.png)',
        '',
        '## Results',
        '',
        '| Image | Input A | Generated B | Target B | PSNR |',
        '| --- | --- | --- | --- | ---: |',
    ]
    for row in rows:
        target = f'![target]({relative_link(row["target"])})' if row['target'] else '-'
        score = f'{row["psnr"]:.4f} dB' if row['psnr'] is not None else '-'
        lines.append(
            f'| `{row["name"]}` | ![input]({relative_link(row["input"])}) '
            f'| ![output]({relative_link(row["output"])}) | {target} | {score} |'
        )
    (output_dir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


@torch.inference_mode()
def test(generator, source_dir, target_dir, output_dir, device, max_images=8,
         checkpoint_path='', epoch=None):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir) if target_dir else None
    output_dir = Path(output_dir)
    image_dir = output_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)

    source_paths = sorted(source_dir.glob('*.png'))
    if max_images > 0:
        source_paths = source_paths[:max_images]
    if not source_paths:
        raise FileNotFoundError(f'No PNG images found in {source_dir}')

    generator.eval()
    rows = []
    for index, source_path in enumerate(source_paths, start=1):
        source = load_image(source_path).unsqueeze(0).to(device)
        padded, height, width = pad_to_multiple(source)
        output = generator(padded)[:, :, :height, :width]

        stem = f'{index:03d}_{source_path.stem}'
        input_path = image_dir / f'{stem}_input.png'
        output_path = image_dir / f'{stem}_output.png'
        save_tensor_image(source[0], input_path)
        save_tensor_image(output[0], output_path)

        target_path = target_dir / source_path.name if target_dir else None
        saved_target_path = None
        score = None
        if target_path and target_path.is_file():
            target = load_image(target_path).unsqueeze(0).to(device)
            if target.shape == output.shape:
                saved_target_path = image_dir / f'{stem}_target.png'
                save_tensor_image(target[0], saved_target_path)
                score = calculate_psnr(output, target)
        rows.append({
            'name': source_path.name,
            'input': input_path,
            'output': output_path,
            'target': saved_target_path,
            'psnr': score,
        })

    paired_scores = [row['psnr'] for row in rows if row['psnr'] is not None]
    mean_psnr = sum(paired_scores) / len(paired_scores) if paired_scores else None
    create_overview(rows, output_dir / 'overview.png')
    write_markdown_report(
        output_dir, checkpoint_path, epoch, rows, mean_psnr)
    return {'images': len(rows), 'mean_psnr': mean_psnr,
            'report_path': output_dir / 'report.md'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run CUT inference and generate a Markdown visualization report')
    parser.add_argument(
        'experiment_name',
        help='experiment directory name under results/ckpt')
    parser.add_argument('--data-root', default=config.DATA_ROOT)
    parser.add_argument('--max-images', type=int, default=8,
                        help='maximum images to render; use 0 for all images')
    parser.add_argument('--device', default=config.DEVICE)
    parser.add_argument('--ngf', type=int, default=config.NGF)
    parser.add_argument('--encoder-blocks', type=int, default=config.ENCODER_BLOCKS)
    parser.add_argument('--decoder-blocks', type=int, default=config.DECODER_BLOCKS)
    parser.add_argument('--downsample-mode', default=config.DOWN_SAMPLE_MODE,
                        choices=('stride', 'avgpool', 'antialias'))
    parser.add_argument('--upsample-mode', default=config.UP_SAMPLE_MODE,
                        choices=('nearest', 'bilinear', 'antialias', 'transposed'))
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
    device = torch.device(args.device)
    checkpoint_path = Path('results/ckpt') / args.experiment_name / 'best_test_psnr.pt'
    output_dir = Path('results/test_visualization') / args.experiment_name
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    generator = CUTGenerator(
        ngf=args.ngf,
        encoder_blocks=args.encoder_blocks,
        decoder_blocks=args.decoder_blocks,
        downsample_mode=args.downsample_mode,
        upsample_mode=args.upsample_mode,
    ).to(device)
    generator.load_state_dict(checkpoint['generator'])
    result = test(
        generator,
        Path(args.data_root) / 'testA',
        Path(args.data_root) / 'testB',
        output_dir,
        device,
        args.max_images,
        checkpoint_path,
        checkpoint.get('epoch'),
    )
    score = (f", mean PSNR={result['mean_psnr']:.4f} dB"
             if result['mean_psnr'] is not None else '')
    print(f"Rendered {result['images']} images{score}")
    print(f"Markdown report: {result['report_path']}")


if __name__ == '__main__':
    main()