"""Create visual reports with PatchNCE as the primary analysis space."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


MODELS = ('cut_baseline', 'expander_decorr_200_exp1_s1407')
MODEL_LABELS = {'cut_baseline': 'CUT', 'expander_decorr_200_exp1_s1407': 'VCReg'}
LAYERS = ('stem', 'downsample_1', 'downsample_2', 'enc_resblock_1', 'enc_resblock_6')
COLORS = {'cut_baseline': '#2f5d8c', 'expander_decorr_200_exp1_s1407': '#d97745'}


def load_nce(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    return {
        model: {
            'ab': [data['mean_nce_embedding_distance'][layer] for layer in LAYERS],
            'ag': [data['mean_nce_source_to_generated_distance'][layer] for layer in LAYERS],
            'bg': [data['mean_nce_clear_to_generated_distance'][layer] for layer in LAYERS],
            'preference': [data['mean_nce_source_preference'][layer] for layer in LAYERS],
        }
        for model in MODELS
    }


def load_raw(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    return {
        model: {
            'ab': [data[model][layer]['ab_cosine_distance']['mean'] for layer in LAYERS],
            'ag': [data[model][layer]['ag_cosine_distance']['mean'] for layer in LAYERS],
            'bg': [data[model][layer]['bg_cosine_distance']['mean'] for layer in LAYERS],
            'preference': [data[model][layer]['source_preference']['mean'] for layer in LAYERS],
        }
        for model in MODELS
    }


def set_axis_style(axis, title, ylabel):
    axis.set_title(title, loc='left', fontsize=12, fontweight='bold')
    axis.set_ylabel(ylabel)
    axis.set_xticks(np.arange(len(LAYERS)), LAYERS, rotation=20, ha='right')
    axis.grid(axis='y', alpha=0.25)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right']].set_visible(False)


def grouped_bars(axis, values, title, ylabel):
    positions = np.arange(len(LAYERS))
    width = 0.36
    for index, model in enumerate(MODELS):
        axis.bar(
            positions + (index - 0.5) * width,
            values[model],
            width,
            label=MODEL_LABELS[model],
            color=COLORS[model],
        )
    set_axis_style(axis, title, ylabel)
    axis.legend(frameon=False)


def line_plot(axis, values, title, ylabel):
    positions = np.arange(len(LAYERS))
    for model in MODELS:
        axis.plot(
            positions,
            values[model],
            marker='o',
            linewidth=2.2,
            label=MODEL_LABELS[model],
            color=COLORS[model],
        )
    axis.axhline(0, color='#555555', linewidth=0.8)
    set_axis_style(axis, title, ylabel)
    axis.legend(frameon=False)


def write_report(path, output_dir, nce, raw):
    lines = [
        '# Encoder Domain Alignment 可视化报告', '',
        '> 主分析空间：PatchNCE/Expander。raw Encoder 仅作为 additional evidence。', '',
        '本报告使用 304 对 `testA/testB` 图像，比较 CUT 与 VCReg 的同位置 patch 几何关系。',
        '主结论应以 NCE 图表为准，因为训练中的 PatchNCE 约束直接作用于 Expander 后的 embedding。', '',
        '## PatchNCE 主分析', '',
        '![PatchNCE 三组距离](nce_distances.png)', '',
        '![PatchNCE source preference](nce_source_preference.png)', '',
        'NCE 空间中，$G(A)$ 对两个模型都更接近 $A$ 而不是 $B$。VCReg 的 source preference',
        '在各层均低于 CUT，说明 VCReg 缩小了生成结果与清晰域之间的 NCE 几何缺口，但没有完全消除源域锚定。', '',
        '## Raw Encoder Additional Evidence', '',
        '![Raw Encoder 三组距离](raw_encoder_distances.png)', '',
        '![Raw Encoder source preference](raw_encoder_source_preference.png)', '',
        'raw Encoder 图用于回答域关系在 Expander 之前是否已经出现。它与 NCE 空间给出一致的定性方向，',
        '但两种空间的绝对距离不可直接比较，因为 Expander 改变了特征坐标和归一化后的几何结构。', '',
        '## 图表数据', '',
        '- NCE 数据：[CUT metrics.json](../../cut_baseline/RRSHID/metrics.json) 和 [VCReg metrics.json](../../expander_decorr_200_exp1_s1407/RRSHID/metrics.json)',
        '- raw Encoder 数据：[summary.json](../full/summary.json)',
        '- 生成脚本：[visualize_encoder_domain_alignment.py](../../../../visualize_encoder_domain_alignment.py)',
        '',
        '图表中的柱状图为 304 对图像的均值；逐图指标仍保留在原始 JSON/JSONL 文件中。',
    ]
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nce-dir', type=Path, default=Path('results/feature_analysis'))
    parser.add_argument('--raw-summary', type=Path, default=Path(
        'results/feature_analysis/encoder_domain_alignment/full/summary.json'))
    parser.add_argument('--output-dir', type=Path, default=Path(
        'results/feature_analysis/encoder_domain_alignment/visual_report'))
    args = parser.parse_args()
    nce = load_nce(args.nce_dir / 'cut_baseline/RRSHID/metrics.json')
    vcreg_nce = load_nce(args.nce_dir / 'expander_decorr_200_exp1_s1407/RRSHID/metrics.json')
    nce['expander_decorr_200_exp1_s1407'] = vcreg_nce['expander_decorr_200_exp1_s1407']
    raw = load_raw(args.raw_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    grouped_bars(axes[0], {model: nce[model]['ab'] for model in MODELS},
                 'Input domain gap', 'Cosine distance')
    grouped_bars(axes[1], {model: nce[model]['ag'] for model in MODELS},
                 'Source to generated', 'Cosine distance')
    grouped_bars(axes[2], {model: nce[model]['bg'] for model in MODELS},
                 'Clear target to generated', 'Cosine distance')
    figure.suptitle('PatchNCE / Expander space: primary domain alignment analysis',
                    fontsize=15, fontweight='bold')
    figure.savefig(args.output_dir / 'nce_distances.png', dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    line_plot(axis, {model: nce[model]['preference'] for model in MODELS},
              'PatchNCE source preference', 'd(B, G(A)) - d(A, G(A))')
    figure.savefig(args.output_dir / 'nce_source_preference.png', dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    grouped_bars(axes[0], {model: raw[model]['ab'] for model in MODELS},
                 'Raw Encoder input gap', 'Cosine distance')
    grouped_bars(axes[1], {model: raw[model]['ag'] for model in MODELS},
                 'Raw Encoder source to generated', 'Cosine distance')
    grouped_bars(axes[2], {model: raw[model]['bg'] for model in MODELS},
                 'Raw Encoder clear target to generated', 'Cosine distance')
    figure.suptitle('Raw Encoder space: additional evidence', fontsize=15, fontweight='bold')
    figure.savefig(args.output_dir / 'raw_encoder_distances.png', dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    line_plot(axis, {model: raw[model]['preference'] for model in MODELS},
              'Raw Encoder source preference', 'd(B, G(A)) - d(A, G(A))')
    figure.savefig(args.output_dir / 'raw_encoder_source_preference.png', dpi=180)
    plt.close(figure)

    write_report(args.output_dir / 'report.md', args.output_dir, nce, raw)
    print(f'Visual report: {args.output_dir / "report.md"}')


if __name__ == '__main__':
    main()