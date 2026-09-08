"""Restyle the selected Figure 3 without changing any plotted observations.

Usage: python build_academic_revision_figure.py
Bundled JSON inputs reproduce the current paper's selected post-hoc weighting.
Only internal accuracy and call rate are reweighted; timing remains original.
No models, APIs, sampling, or new evaluations are involved.
"""
import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ['never', 'conservative', 'balanced', 'aggressive', 'always']
LABELS = ['L', 'C', 'B', 'A', 'W']
POOLS = [('frozen', 'Internal'), ('striviaqa', 'Speech TriviaQA'), ('sdqa', 'SD-QA')]
INK, BLUE, GRAY, LIGHT, RUST = '#252525', '#29475F', '#777777', '#D9D9D9', '#885D46'


def make_figure(native_path, weights_path, output_dir):
    native = json.loads(native_path.read_text())
    original = native['pools']
    data = copy.deepcopy(original)
    weights = json.loads(weights_path.read_text())
    selected = next(item for item in weights['all_sixteen_joint_scenarios']
                    if item['knowledge_weight'] == .25 and item['math_weight'] == .25)
    for arm in ARMS:
        data['frozen'][arm]['accuracy'] = selected['arms'][arm]['accuracy']
        data['frozen'][arm]['rate'] = selected['arms'][arm]['call_rate']
    # Explicit invariants: the selection affects only two fields of one pool.
    for pool, _ in POOLS:
        for arm in ARMS:
            for key, value in original[pool][arm].items():
                if pool == 'frozen' and key in ('accuracy', 'rate'):
                    continue
                assert data[pool][arm][key] == value, (pool, arm, key)

    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['Times New Roman', 'STIXGeneral'],
        'font.size': 8.6, 'mathtext.fontset': 'stix',
        'axes.labelsize': 8.8, 'axes.titlesize': 9.2,
        'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
        'axes.linewidth': .6, 'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.color': INK, 'ytick.color': INK,
        'xtick.labelsize': 8.1, 'ytick.labelsize': 8.1,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'xtick.major.width': .6, 'ytick.major.width': .6,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'legend.fontsize': 7.8, 'legend.frameon': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 6.35),
                             gridspec_kw={'hspace': .42, 'wspace': .33})
    fig.subplots_adjust(left=.094, right=.983, top=.918, bottom=.123)
    marker_kw = dict(markersize=3.5, markerfacecolor='white', markeredgewidth=.85)
    drawn = {}
    for row, (pool, title) in enumerate(POOLS):
        p = data[pool]
        ax, tx = axes[row]
        x = np.array([p[a]['rate'] * 100 for a in ARMS])
        y = np.array([p[a]['accuracy'] * 100 for a in ARMS])
        for panel in (ax, tx):
            panel.set_axisbelow(True)
            panel.grid(axis='y', color=LIGHT, linewidth=.35, alpha=.75)
            panel.tick_params(axis='both', pad=3)
        random_line, = ax.plot([0, 100], [y[0], y[-1]], color=GRAY,
                               linestyle=(0, (4, 2.4)), linewidth=.95, zorder=2)
        always_line = ax.axhline(y[-1], color=INK, linestyle=(0, (1, 2.2)),
                                 linewidth=.8, zorder=2)
        gate_line, = ax.plot(x, y, color=BLUE, linewidth=1.3, marker='o',
                             zorder=4, **marker_kw)
        ax.plot(x[-1], y[-1], color=BLUE, marker='s', linestyle='None',
                zorder=5, **marker_kw)
        for j, label in enumerate(LABELS):
            # Keep short arm labels, in neutral text, clear of adjacent points.
            offsets = [(3, -12), (3, 6), (3, 6), (-10, -8), (-10, -12)]
            if pool == 'striviaqa':
                offsets[0], offsets[1] = (3, 7), (3, -12)
            offset = offsets[j]
            ax.annotate(label, (x[j], y[j]), xytext=offset,
                        textcoords='offset points', fontsize=7.5, color=INK)
        ax.set(xlim=(-3, 105), ylim=(30, 104),
               xticks=[0, 25, 50, 75, 100], yticks=[40, 60, 80, 100])
        ax.set_ylabel('Weighted accuracy (%)' if row == 0 else 'Accuracy (%)', labelpad=6)
        ax.set_title(f'{title}  ($n={p["never"]["n"]}$)', loc='left', pad=5,
                     fontweight='normal')
        timing_lines = []
        for key, lab, color, line_style, marker in [
                ('mean_s', 'Mean', BLUE, '-', 'o'),
                ('p50_s', 'P50', GRAY, (0, (4, 2.4)), 's'),
                ('p95_s', 'P95', RUST, (0, (1, 2.2)), '^')]:
            values = [p[a][key] for a in ARMS]
            line, = tx.plot(range(5), values, label=lab, color=color,
                            linestyle=line_style, linewidth=1.05, **marker_kw,
                            marker=marker)
            timing_lines.append(line)
            assert np.array_equal(line.get_ydata(), np.array([original[pool][a][key] for a in ARMS]))
        tx.set_yscale('log')
        tx.set(ylim=(1, 350), xlim=(-.15, 4.15), xticks=range(5),
               xticklabels=LABELS, yticks=[1, 3, 10, 30, 100, 300],
               yticklabels=['1', '3', '10', '30', '100', '300'])
        tx.minorticks_off()
        tx.set_ylabel('Reconstructed time (s)', labelpad=6)
        tx.set_title('Timing diagnostic', loc='left', pad=5, fontweight='normal')
        if row == 2:
            ax.set_xlabel('Realized expert call rate (%)', labelpad=5)
            tx.set_xlabel('Recorded arm', labelpad=5)
        if row == 0:
            centers = [sum(panel.get_position().intervalx) / 2 for panel in (ax, tx)]
            fig.legend([gate_line, random_line, always_line],
                       ['Gate', 'Random mixture', 'Always'], ncol=3,
                       loc='upper center', bbox_to_anchor=(centers[0], .993),
                       handlelength=2.1, handletextpad=.5, columnspacing=1.3)
            fig.legend(timing_lines, ['Mean', 'P50', 'P95'], ncol=3,
                       loc='upper center', bbox_to_anchor=(centers[1], .993),
                       handlelength=2.1, handletextpad=.5, columnspacing=1.5)
        assert np.array_equal(gate_line.get_xdata(), x)
        assert np.array_equal(gate_line.get_ydata(), y)
        drawn[pool] = {'accuracy_x': x.tolist(), 'accuracy_y': y.tolist(),
                       'random_x': [0, 100], 'random_y': [float(y[0]), float(y[-1])],
                       'always_reference': float(y[-1]),
                       'timing': {key: [p[a][key] for a in ARMS]
                                  for key in ('mean_s', 'p50_s', 'p95_s')}}

    fig.text(.094, .031,
             'Internal accuracy: post-hoc weights of 0.25 for knowledge and math, and 1 for other categories.',
             fontsize=6.7, va='bottom', style='italic')
    fig.text(.094, .011, 'Internal timing uses the original mixture.',
             fontsize=6.7, va='bottom', style='italic')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'revision_accuracy_latency.pdf')
    fig.savefig(output_dir / 'revision_accuracy_latency.png', dpi=220)
    plt.close(fig)
    (output_dir / 'plotted_values.json').write_text(json.dumps(drawn, indent=2) + '\n')
    return drawn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-summary', type=Path, default=HERE / 'revision_data/native_summary.json')
    parser.add_argument('--reweighting', type=Path, default=HERE / 'revision_data/joint_reweighting_ideas.json')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'figures')
    args = parser.parse_args()
    make_figure(args.native_summary, args.reweighting, args.output_dir)
