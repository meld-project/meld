#!/usr/bin/env python3
"""
论文图表绘制脚本
生成所有论文需要的图
"""
import json
import os
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager as fm

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Songti SC', 'STSong', 'SimSun', 'Nimbus Roman', 'TeX Gyre Termes', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

CN_FONT = fm.FontProperties(family='Songti SC')

RESULTS_DIR = Path(os.environ.get("MELD_RESULTS_DIR", "experiments/results"))
FIGURES_DIR = Path(os.environ.get("MELD_FIGURES_DIR", str(RESULTS_DIR / "figures")))
FIGURES_DIR.mkdir(exist_ok=True)

COLORS = {
    'Qwen3-0.6B': '#e41a1c',
    'Qwen2.5-0.5B': '#ff7f00',
    'LLaMA-2-7B': '#377eb8',
    'BERT-base': '#4daf4a',
    'GTE-large': '#984ea3',
}
MARKERS = {
    'Qwen3-0.6B': 'o',
    'Qwen2.5-0.5B': 's',
    'LLaMA-2-7B': '^',
    'BERT-base': 'D',
    'GTE-large': 'v',
}

MODEL_FILES = {
    'Qwen3-0.6B': 'qwen3_layer_profile.json',
    'Qwen2.5-0.5B': 'qwen2.5_layer_profile.json',
    'LLaMA-2-7B': 'llama2_layer_profile.json',
    'BERT-base': 'bert_layer_profile.json',
    'GTE-large': 'gte_large_layer_profile.json',
}

FAMILY_OOD_FILES = {
    'Qwen3-0.6B': 'qwen3_family_ood_layer_profile.json',
    'Qwen2.5-0.5B': 'qwen2.5_family_ood_layer_profile.json',
    'LLaMA-2-7B': 'llama2_family_ood_layer_profile.json',
    'BERT-base': 'bert_family_ood_layer_profile.json',
    'GTE-large': 'gte_large_family_ood_layer_profile.json',
}


def load_layer_profile(name):
    with open(RESULTS_DIR / MODEL_FILES[name]) as f:
        return json.load(f)


def load_family_ood(name):
    with open(RESULTS_DIR / FAMILY_OOD_FILES[name]) as f:
        return json.load(f)


# ============================================================
# 图1: 逐层 TPR@0.1%FPR 曲线（5模型，归一化x轴）
# ============================================================
def plot_layer_tpr_curves():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for name in MODEL_FILES:
        d = load_layer_profile(name)
        layers = d['layers']
        num = len(layers)
        x = [(l['layer'] / num) * 100 for l in layers]
        y = [l['tpr_at_fpr_0.001'] for l in layers]
        ax.plot(x, y, color=COLORS[name], marker=MARKERS[name],
                markersize=4, linewidth=1.5, label=name, alpha=0.85)

    ax.set_xlabel('相对层位置（%）', fontproperties=CN_FONT)
    ax.set_ylabel('TPR@0.1%FPR', fontproperties=CN_FONT)
    ax.set_title('5个模型逐层TPR@0.1%FPR曲线', fontproperties=CN_FONT)
    ax.legend(loc='lower left', prop=CN_FONT)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    fig.savefig(FIGURES_DIR / 'fig_layer_tpr_curves.pdf')
    fig.savefig(FIGURES_DIR / 'fig_layer_tpr_curves.png')
    plt.close(fig)
    print("图1: 逐层TPR曲线 -> fig_layer_tpr_curves.pdf")


# ============================================================
# 图2: 逐层 Macro-F1 曲线（5模型）
# ============================================================
def plot_layer_f1_curves():
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for name in MODEL_FILES:
        d = load_layer_profile(name)
        layers = d['layers']
        num = len(layers)
        x = [(l['layer'] / num) * 100 for l in layers]
        y = [l['macro_f1'] for l in layers]
        ax.plot(x, y, color=COLORS[name], marker=MARKERS[name],
                markersize=4, linewidth=1.5, label=name, alpha=0.85)

    ax.set_xlabel('相对层位置（%）', fontproperties=CN_FONT)
    ax.set_ylabel('宏平均F1', fontproperties=CN_FONT)
    ax.set_title('5个模型逐层宏平均F1曲线', fontproperties=CN_FONT)
    ax.legend(loc='lower left', prop=CN_FONT)
    ax.set_xlim(0, 105)
    ax.grid(True, alpha=0.3)

    fig.savefig(FIGURES_DIR / 'fig_layer_f1_curves.pdf')
    fig.savefig(FIGURES_DIR / 'fig_layer_f1_curves.png')
    plt.close(fig)
    print("图2: 逐层F1曲线 -> fig_layer_f1_curves.pdf")


# ============================================================
# 图3: Family-OOD 最优层分布箱线图
# ============================================================
def plot_family_ood_best_layers():
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    data = []
    labels = []
    for name in FAMILY_OOD_FILES:
        d = load_family_ood(name)
        stability = d['stability']
        # 归一化到百分比
        num_layers_map = {'Qwen3-0.6B': 28, 'Qwen2.5-0.5B': 24, 'LLaMA-2-7B': 32, 'BERT-base': 12, 'GTE-large': 24}
        nl = num_layers_map[name]
        normalized = [l / nl * 100 for l in stability['best_layers']]
        data.append(normalized)
        labels.append(name)

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.6)
    for i, (patch, name) in enumerate(zip(bp['boxes'], labels)):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.6)

    # 最终层位置标线
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='最终层（100%）')

    ax.set_ylabel('最佳层位置（%）', fontproperties=CN_FONT)
    ax.set_title('Family-OOD最佳层位置分布', fontproperties=CN_FONT)
    ax.legend(prop=CN_FONT)
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')

    fig.savefig(FIGURES_DIR / 'fig_family_ood_best_layers.pdf')
    fig.savefig(FIGURES_DIR / 'fig_family_ood_best_layers.png')
    plt.close(fig)
    print("图3: Family-OOD最优层箱线图 -> fig_family_ood_best_layers.pdf")


# ============================================================
# 图4: Hash消融对比
# ============================================================
def plot_hash_ablation():
    try:
        with open(RESULTS_DIR / 'hash_ablation' / 'hash_ablation_results.json') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("图4: Hash消融数据不存在，跳过")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    layers = ['选层\n（L7）', '最终层\n（L28）']
    with_hash = [
        data['with_hash']['layer_7']['tpr_at_fpr_0.001'],
        data['with_hash']['layer_28']['tpr_at_fpr_0.001'],
    ]
    without_hash = [
        data['without_hash']['layer_7']['tpr_at_fpr_0.001'],
        data['without_hash']['layer_28']['tpr_at_fpr_0.001'],
    ]

    x = np.arange(len(layers))
    width = 0.3

    bars1 = ax.bar(x - width/2, with_hash, width, label='含哈希', color='#377eb8', alpha=0.8)
    bars2 = ax.bar(x + width/2, without_hash, width, label='去哈希', color='#e41a1c', alpha=0.8)

    # 标注差异
    for i in range(len(layers)):
        diff = with_hash[i] - without_hash[i]
        y_pos = max(with_hash[i], without_hash[i]) + 0.02
        ax.annotate(f'-{diff:.1%}', xy=(x[i], y_pos), ha='center', fontsize=10, fontweight='bold')

    ax.set_ylabel('TPR@0.1%FPR', fontproperties=CN_FONT)
    ax.set_title('哈希消融对检测性能的影响', fontproperties=CN_FONT)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, fontproperties=CN_FONT)
    ax.legend(prop=CN_FONT)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    fig.savefig(FIGURES_DIR / 'fig_hash_ablation.pdf')
    fig.savefig(FIGURES_DIR / 'fig_hash_ablation.png')
    plt.close(fig)
    print("图4: Hash消融 -> fig_hash_ablation.pdf")


# ============================================================
# 图5: Fisher判别比逐层
# ============================================================
def plot_fisher_ratios():
    try:
        with open(RESULTS_DIR / 'analysis' / 'fisher_ratios.json') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("图5: Fisher数据不存在，跳过")
        return

    layers = [l['layer'] for l in data['layers']]
    fisher = [l['fisher_ratio'] for l in data['layers']]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(layers, fisher, color='#377eb8', alpha=0.7, edgecolor='#2c6fad')

    # 标注峰值
    peak_idx = np.argmax(fisher)
    ax.annotate(f'峰值：L{layers[peak_idx]}\n({fisher[peak_idx]:.0f})',
                xy=(layers[peak_idx], fisher[peak_idx]),
                xytext=(layers[peak_idx]+3, fisher[peak_idx]+20),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red', fontproperties=CN_FONT)

    ax.set_xlabel('层号', fontproperties=CN_FONT)
    ax.set_ylabel('Fisher判别比', fontproperties=CN_FONT)
    ax.set_title('Qwen3-0.6B逐层Fisher判别比', fontproperties=CN_FONT)
    ax.grid(True, alpha=0.3, axis='y')

    fig.savefig(FIGURES_DIR / 'fig_fisher_ratios.pdf')
    fig.savefig(FIGURES_DIR / 'fig_fisher_ratios.png')
    plt.close(fig)
    print("图5: Fisher判别比 -> fig_fisher_ratios.pdf")


# ============================================================
# 图6: t-SNE可视化（3个子图：浅层/中间层/最终层）
# ============================================================
def plot_tsne():
    tsne_dir = RESULTS_DIR / 'analysis' / 'tsne'
    if not tsne_dir.exists():
        print("图6: t-SNE数据不存在，跳过")
        return

    tsne_files = sorted(tsne_dir.glob('tsne_layer_*.npz'))
    if len(tsne_files) < 3:
        print(f"图6: 只有{len(tsne_files)}个t-SNE文件，需要3个")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = ['浅层（L3）', '中间层（L18）', '最终层（L28）']

    for i, (fpath, title) in enumerate(zip(tsne_files, titles)):
        data = np.load(fpath)
        coords = data['coords']
        labels = data['labels']

        mal_mask = labels == 1
        ben_mask = labels == 0

        ax = axes[i]
        ax.scatter(coords[ben_mask, 0], coords[ben_mask, 1],
                   c='#4daf4a', s=3, alpha=0.3, label='良性')
        ax.scatter(coords[mal_mask, 0], coords[mal_mask, 1],
                   c='#e41a1c', s=3, alpha=0.3, label='恶意')
        ax.set_title(title, fontproperties=CN_FONT)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.legend(markerscale=5, loc='upper left', prop=CN_FONT)

    fig.suptitle('Qwen3-0.6B不同层表示的t-SNE可视化', y=1.02, fontproperties=CN_FONT)
    fig.savefig(FIGURES_DIR / 'fig_tsne_layers.pdf')
    fig.savefig(FIGURES_DIR / 'fig_tsne_layers.png')
    plt.close(fig)
    print("图6: t-SNE -> fig_tsne_layers.pdf")


# ============================================================
# 图7: Time-OOD 方法对比柱状图
# ============================================================
def plot_method_comparison():
    qwen3 = load_layer_profile('Qwen3-0.6B')
    gte = load_layer_profile('GTE-large')
    qwen3_best = max(qwen3['layers'], key=lambda x: x['tpr_at_fpr_0.001'])
    qwen3_final = qwen3['layers'][-1]
    gte_best = max(gte['layers'], key=lambda x: x['tpr_at_fpr_0.001'])

    methods = [
        ('Qwen3微调', 0.9979),
        ('BERT微调', 0.9958),
        (f'MELD选层\n（L{qwen3_best["layer"]}）', qwen3_best['tpr_at_fpr_0.001']),
        ('MELD-MLP', 0.9752),
        ('MELD+MLP', 0.9752),
        ('TF-IDF词\n+GBDT', 0.9094),
        ('MELD均值池化', 0.9201),
        ('MELD末4层', 0.8991),
        ('TF-IDF词\n+LR', 0.8651),
        (f'MELD最终层\n（L{qwen3_final["layer"]}）', qwen3_final['tpr_at_fpr_0.001']),
        (f'GTE-large选层\n（L{gte_best["layer"]}）', gte_best['tpr_at_fpr_0.001']),
    ]

    names = [m[0] for m in methods]
    values = [m[1] for m in methods]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    colors = []
    for n in names:
        if '微调' in n:
            colors.append('#ff7f00')  # 微调=橙
        elif 'GTE-large' in n:
            colors.append('#984ea3')  # 专用嵌入=紫
        elif 'MELD' in n:
            colors.append('#377eb8')  # MELD=蓝
        else:
            colors.append('#4daf4a')  # 基线=绿

    bars = ax.bar(range(len(names)), values, color=colors, alpha=0.8, edgecolor='gray')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8.5, fontproperties=CN_FONT)
    ax.set_ylabel('TPR@0.1%FPR', fontproperties=CN_FONT)
    ax.set_title('Time-OOD方法对比（TPR@0.1%FPR）', fontproperties=CN_FONT)
    ax.set_ylim(0.8, 1.02)
    ax.grid(True, alpha=0.3, axis='y')

    # 数值标注
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#ff7f00', alpha=0.8, label='微调方法'),
        Patch(facecolor='#377eb8', alpha=0.8, label='MELD冻结变体'),
        Patch(facecolor='#984ea3', alpha=0.8, label='专用嵌入模型'),
        Patch(facecolor='#4daf4a', alpha=0.8, label='传统基线'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', prop=CN_FONT)

    fig.savefig(FIGURES_DIR / 'fig_method_comparison.pdf')
    fig.savefig(FIGURES_DIR / 'fig_method_comparison.png')
    plt.close(fig)
    print("图7: 方法对比柱状图 -> fig_method_comparison.pdf")


# ============================================================
# 图8: Qwen3 逐层完整景观（TPR@0.1% + F1 双轴）
# ============================================================
def plot_qwen3_dual_axis():
    d = load_layer_profile('Qwen3-0.6B')
    layers_data = d['layers']
    x = [l['layer'] for l in layers_data]
    tpr = [l['tpr_at_fpr_0.001'] for l in layers_data]
    f1 = [l['macro_f1'] for l in layers_data]

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
    ax2 = ax1.twinx()

    line1 = ax1.plot(x, tpr, 'o-', color='#e41a1c', markersize=5, linewidth=2, label='TPR@0.1%FPR')
    line2 = ax2.plot(x, f1, 's--', color='#377eb8', markersize=4, linewidth=1.5, alpha=0.7, label='宏平均F1')

    ax1.set_xlabel('层号', fontproperties=CN_FONT)
    ax1.set_ylabel('TPR@0.1%FPR', color='#e41a1c', fontproperties=CN_FONT)
    ax2.set_ylabel('宏平均F1', color='#377eb8', fontproperties=CN_FONT)
    ax1.set_title('Qwen3-0.6B逐层性能剖面', fontproperties=CN_FONT)

    # 标注最优层和最终层
    best_tpr_idx = np.argmax(tpr)
    ax1.annotate(f'最优：L{x[best_tpr_idx]}\nTPR={tpr[best_tpr_idx]:.4f}',
                 xy=(x[best_tpr_idx], tpr[best_tpr_idx]),
                 xytext=(x[best_tpr_idx]+4, tpr[best_tpr_idx]-0.05),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=9, color='red', fontproperties=CN_FONT)
    ax1.annotate(f'最终层：L{x[-1]}\nTPR={tpr[-1]:.4f}',
                 xy=(x[-1], tpr[-1]),
                 xytext=(x[-1]-6, tpr[-1]-0.08),
                 arrowprops=dict(arrowstyle='->', color='darkred'),
                 fontsize=9, color='darkred', fontproperties=CN_FONT)

    lines = line1 + line2
    labs = [l.get_label() for l in lines]
    ax1.legend(lines, labs, loc='lower left', prop=CN_FONT)
    ax1.grid(True, alpha=0.3)

    fig.savefig(FIGURES_DIR / 'fig_qwen3_dual_axis.pdf')
    fig.savefig(FIGURES_DIR / 'fig_qwen3_dual_axis.png')
    plt.close(fig)
    print("图8: Qwen3双轴图 -> fig_qwen3_dual_axis.pdf")


# ============================================================
if __name__ == '__main__':
    print("开始绘图...")
    plot_layer_tpr_curves()
    plot_layer_f1_curves()
    plot_family_ood_best_layers()
    plot_hash_ablation()
    plot_fisher_ratios()
    plot_tsne()
    plot_method_comparison()
    plot_qwen3_dual_axis()
    print(f"\n全部图表已保存至 {FIGURES_DIR}/")
