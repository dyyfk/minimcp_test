"""Redraw the layer-transfer figure from archived sweep values, without refitting.

The shaded region locates the observed late-layer decline; it does not assign
a cause to it. Source files are read from the current Git archive when absent
from a sparse checkout. Outputs are vector PDFs with embedded TrueType fonts.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODELS = [
    ('minicpm-o45', 'MiniCPM-o 4.5 (duplex)', '#2a78d6', '-'),
    ('qwen3-8b', 'Qwen3-8B (raw backbone)', '#b35c20', '--'),
    ('minicpm-o26', 'MiniCPM-o 2.6 (duplex)', '#1b9a70', '-'),
    ('qwen2.5-7b', 'Qwen2.5-7B (raw)', '#b78900', '--'),
    ('qwen2.5-omni-7b', 'Qwen2.5-Omni (streaming)', '#b96b91', ':'),
]


def archived_bytes(path):
    local = ROOT / path
    return local.read_bytes() if local.exists() else subprocess.check_output(
        ['git', 'show', 'HEAD:' + path], cwd=ROOT)


def make_layer_figure():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'STIXGeneral'],
        'mathtext.fontset': 'stix', 'font.size': 9, 'axes.titlesize': 9.5,
        'axes.labelsize': 9, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': '#555555', 'axes.linewidth': .6,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    fig.subplots_adjust(left=.083, right=.985, bottom=.20, top=.77, wspace=.12)
    audit = {}
    for tag, label, color, ls in MODELS:
        path = f'interactive_paper/data/layers/layer_sweep_{tag}.json'
        raw = archived_bytes(path)
        data = json.loads(raw)
        x = [(c['layer'] + 1) / data['n_layers'] for c in data['curves']]
        curves = {}
        for ax, read in zip(axes, ['last', 'mean']):
            y = [c[f'{read}_lopo_hard-math'] for c in data['curves']]
            ax.plot(x, y, color=color, ls=ls, lw=1.8 if tag == 'minicpm-o45' else 1.2,
                    label=label, zorder=3)
            curves[read] = y
        audit[tag] = {'source': path, 'sha256': hashlib.sha256(raw).hexdigest(),
                      'relative_depth': x, **curves}
    for ax, title in zip(axes, ['Last-token read', 'Mean-pooled read']):
        ax.axvspan(.85, 1, color='#2a78d6', alpha=.07, lw=0, zorder=0)
        ax.axhline(.5, color='#888888', lw=.7, ls=(0, (3, 3)))
        ax.set(xlim=(0, 1.02), ylim=(.30, 1), xlabel='Relative layer depth', title=title)
        ax.grid(axis='y', color='#dddddd', lw=.4, zorder=0)
    axes[0].set_ylabel('LOPO math AUC')
    axes[0].axvline(23/36, color='#555555', lw=.8, ls=(0, (1, 2)), zorder=1)
    axes[0].text(23/36-.02, .72, 'L22 readout', rotation=90, ha='right', va='top', fontsize=8)
    axes[0].annotate('Late-layer decline', xy=(1, audit['minicpm-o45']['last'][-1]),
                     xytext=(.29, .405), fontsize=8.5, color='#205da4',
                     arrowprops={'arrowstyle': '->', 'color': '#205da4', 'lw': .9},
                     bbox={'facecolor':'white', 'edgecolor':'none', 'alpha':.9, 'pad':1.5})
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, 1.015),
               ncol=3, frameon=False, columnspacing=1.2, handlelength=2.4)
    fig.text(.083, .018, 'Text-input LOPO math; shading marks the final 15% of layer depth.',
             color='#444444', fontsize=8, style='italic')
    out = HERE/'figures'
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/'layer_sweep.pdf')
    fig.savefig(out/'layer_sweep.png', dpi=220)
    mirror = HERE.parent/'figures'
    mirror.mkdir(parents=True, exist_ok=True)
    for suffix in ('pdf', 'png'):
        shutil.copy2(out/f'layer_sweep.{suffix}', mirror/f'layer_sweep.{suffix}')
    plt.close(fig)
    (HERE/'revision_data/layer_sweep_values.json').write_text(json.dumps(audit, indent=2)+'\n')
    return audit


if __name__ == '__main__':
    make_layer_figure()
