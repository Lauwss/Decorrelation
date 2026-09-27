import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


METRICS = (
    'loss_g', 'loss_g_gan', 'loss_d', 'loss_nce', 'loss_idt_nce',
    'lr_g', 'lr_d', 'epoch_time_seconds', 'train_psnr', 'test_psnr',
)


def optional_float(value):
    if value is None or not value.strip():
        return None
    return float(value)


def load_metrics(log_path):
    rows = []
    with Path(log_path).open(newline='', encoding='utf-8') as log_file:
        for raw_row in csv.DictReader(log_file):
            try:
                row = {'epoch': int(raw_row['epoch'])}
                row.update({name: optional_float(raw_row.get(name)) for name in METRICS})
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f'No valid training rows found in {log_path}')
    return rows


def available_points(rows, metric):
    return [(row['epoch'], row[metric]) for row in rows if row[metric] is not None]


def plot_line(axis, rows, metric, label, **kwargs):
    points = available_points(rows, metric)
    if points:
        epochs, values = zip(*points)
        axis.plot(epochs, values, label=label, **kwargs)
    return points


def create_figure(rows, output_path):
    plt.style.use('seaborn-v0_8-whitegrid')
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    psnr_axis = axes[0, 0]
    plot_line(psnr_axis, rows, 'train_psnr', 'Train PSNR', marker='o', markersize=3)
    test_points = plot_line(
        psnr_axis, rows, 'test_psnr', 'Test PSNR', marker='o', markersize=3)
    if test_points:
        best_epoch, best_psnr = max(test_points, key=lambda point: point[1])
        psnr_axis.scatter(best_epoch, best_psnr, color='#c43d3d', zorder=5)
        psnr_axis.annotate(
            f'Best {best_psnr:.3f} dB\nEpoch {best_epoch}',
            (best_epoch, best_psnr), xytext=(8, 8), textcoords='offset points')
    psnr_axis.set_title('PSNR')
    psnr_axis.set_ylabel('PSNR (dB)')
    psnr_axis.legend()

    loss_axis = axes[0, 1]
    for metric, label in (
        ('loss_g', 'G total'), ('loss_g_gan', 'G GAN'), ('loss_d', 'D'),
        ('loss_nce', 'NCE'), ('loss_idt_nce', 'Identity NCE'),
    ):
        plot_line(loss_axis, rows, metric, label, linewidth=1.3)
    loss_axis.set_title('Training Losses')
    loss_axis.set_ylabel('Loss')
    loss_axis.set_yscale('log')
    loss_axis.legend(ncols=2)

    lr_axis = axes[1, 0]
    plot_line(lr_axis, rows, 'lr_g', 'Generator LR')
    plot_line(lr_axis, rows, 'lr_d', 'Discriminator LR', linestyle='--')
    lr_axis.set_title('Learning Rate')
    lr_axis.set_ylabel('Learning rate')
    lr_axis.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    lr_axis.legend()

    time_axis = axes[1, 1]
    time_points = plot_line(
        time_axis, rows, 'epoch_time_seconds', 'Epoch time', color='#26734d')
    if time_points:
        mean_time = sum(point[1] for point in time_points) / len(time_points)
        time_axis.axhline(
            mean_time, color='#c47a20', linestyle='--',
            label=f'Mean {mean_time:.2f}s')
    time_axis.set_title('Training Throughput')
    time_axis.set_ylabel('Seconds per epoch')
    time_axis.legend()

    for axis in axes.flat:
        axis.set_xlabel('Epoch')
        axis.margins(x=0.02)
    figure.suptitle('CUT Training Curves', fontsize=16)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def format_metric(value, unit=''):
    return f'{value:.4f}{unit}' if value is not None else 'unavailable'


def write_report(rows, log_path, output_dir):
    test_points = available_points(rows, 'test_psnr')
    train_points = available_points(rows, 'train_psnr')
    best_test = max(test_points, key=lambda point: point[1]) if test_points else None
    latest = rows[-1]
    lines = [
        '# CUT Training Curves',
        '',
        f'- Source: `{Path(log_path)}`',
        f'- Epoch range: {rows[0]["epoch"]}-{latest["epoch"]}',
        f'- Logged epochs: {len(rows)}',
        f'- PSNR evaluations: {len(test_points)}',
    ]
    if best_test:
        lines.append(f'- Best test PSNR: {best_test[1]:.4f} dB at epoch {best_test[0]}')
    if train_points:
        best_train = max(train_points, key=lambda point: point[1])
        lines.append(f'- Best train PSNR: {best_train[1]:.4f} dB at epoch {best_train[0]}')
    lines.extend([
        '',
        '![PSNR, losses, learning rate, and epoch time](training_curves.png)',
        '',
        '## Latest Metrics',
        '',
        '| Metric | Value |',
        '| --- | ---: |',
        f'| Epoch | {latest["epoch"]} |',
        f'| Generator loss | {format_metric(latest["loss_g"])} |',
        f'| GAN loss | {format_metric(latest["loss_g_gan"])} |',
        f'| Discriminator loss | {format_metric(latest["loss_d"])} |',
        f'| NCE loss | {format_metric(latest["loss_nce"])} |',
        f'| Identity NCE loss | {format_metric(latest["loss_idt_nce"])} |',
        f'| Train PSNR | {format_metric(latest["train_psnr"], " dB")} |',
        f'| Test PSNR | {format_metric(latest["test_psnr"], " dB")} |',
        f'| Epoch time | {format_metric(latest["epoch_time_seconds"], " s")} |',
    ])
    report_path = output_dir / 'report.md'
    report_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report_path, best_test


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render CUT training metrics as PNG and Markdown')
    parser.add_argument(
        '--log', default='results/ckpt/cut_baseline/train_log.csv',
        help='path to the training CSV log')
    parser.add_argument(
        '--output-dir', default=None,
        help='output directory; defaults to <log directory>/training_visualization')
    return parser.parse_args()


def main():
    args = parse_args()
    log_path = Path(args.log)
    output_dir = (Path(args.output_dir) if args.output_dir else
                  log_path.parent / 'training_visualization')
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_metrics(log_path)
    figure_path = output_dir / 'training_curves.png'
    create_figure(rows, figure_path)
    report_path, best_test = write_report(rows, log_path, output_dir)
    print(f'Rendered {len(rows)} epochs to {figure_path}')
    if best_test:
        print(f'Best test PSNR: {best_test[1]:.4f} dB at epoch {best_test[0]}')
    print(f'Markdown report: {report_path}')


if __name__ == '__main__':
    main()