"""Plot policy Pareto frontiers using retained accuracy and latest measured TTFA.

Usage: python build_academic_revision_figure.py
Internal accuracy, call rate, and timing use the same full 240 query IDs with
unit weights. External curves use their original query pools.
TTFA is the wait after input end; completed early answers count as zero wait.
No models, APIs, sampling, or new evaluations are involved.
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from build_main_results import check_figure, load_figure_pools, pareto_indices, METRIC_SOURCES
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ARMS = ['never', 'conservative', 'balanced', 'aggressive', 'always']
LABELS = ['L', 'C', 'B', 'A', 'W']
POOLS = [('frozen', 'Internal'), ('striviaqa', 'Speech TriviaQA'), ('sdqa', 'SD-QA')]
INK, BLUE, GRAY, LIGHT = '#252525', '#29475F', '#777777', '#D9D9D9'


def make_figure(native_path, output_dir, internal_path=HERE / 'revision_data/internal_unweighted_summary.json',
                ttfa_path=HERE.parent / 'ttfa_real/nonnegative/summary.json'):
    data = load_figure_pools(native_path, internal_path, ttfa_path)

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
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 5.45),
                             gridspec_kw={'hspace': .42, 'wspace': .29})
    fig.subplots_adjust(left=.085, right=.985, top=.885, bottom=.12)
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
        frontier = pareto_indices(x, y)
        ax.plot(x, y, color=BLUE, linewidth=.65, alpha=.65, zorder=2.5)
        gate_line, = ax.plot(x, y, color=BLUE, linewidth=0, marker='o',
                             zorder=4, **marker_kw)
        frontier_line, = ax.plot(x[frontier], y[frontier], color=BLUE, lw=1.35,
                                zorder=3, label='Pareto frontier')
        ax.plot(x[-1], y[-1], color='#b35c20', marker='s', linestyle='None',
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
        ax.set_ylabel('Accuracy (%)', labelpad=6)
        ax.set_title(f'{title}  ($n={p["never"]["n"]}$)', loc='left', pad=5,
                     fontweight='normal')
        latency = np.array([p[a]['mean_s'] for a in ARMS])
        # The reporting convention retains judged accuracy and takes all TTFA
        # values from the full timing run. Dominance is over policy summaries.
        latency_frontier = sorted(pareto_indices(latency, y), key=lambda i: latency[i])
        tx.plot(latency, y, color=BLUE, linewidth=.65, alpha=.65, zorder=2.5)
        latency_line, = tx.plot(latency[latency_frontier], y[latency_frontier],
                                color=BLUE, lw=1.35, zorder=3)
        colors = [GRAY, BLUE, BLUE, BLUE, '#b35c20']
        for j, label in enumerate(LABELS):
            tx.plot(latency[j], y[j], linestyle='None', color=colors[j],
                    marker='s' if j == 4 else 'o', **marker_kw, zorder=4)
            offset = (4, 6)
            if j == 0: offset = (-4, -12)
            if j == 4: offset = (-12, -12)
            if pool == 'striviaqa' and j == 1: offset = (4, -11)
            if pool == 'striviaqa' and j == 2: offset = (-10, 8)
            tx.annotate(label, (latency[j], y[j]), xytext=offset,
                        textcoords='offset points', fontsize=7.5, color=INK)
        tx.set(ylim=(30, 104), xlim=(0, 10.5),
               xticks=[0, 2, 4, 6, 8, 10], yticks=[40, 60, 80, 100])
        tx.minorticks_off()
        tx.set_ylabel('Accuracy (%)', labelpad=6)
        tx.set_title('Accuracy vs. latency', loc='left', pad=5, fontsize=8.6)
        if pool == 'frozen':
            tx.text(.98, .08, 'W: always-escalate\nvia GPT-5.5', transform=tx.transAxes,
                    ha='right', va='bottom', color='#8a471b', fontsize=7.5)
        if row == 2:
            ax.set_xlabel('Realized expert call rate (%)', labelpad=5)
            tx.set_xlabel('Mean server TTFA (s)', labelpad=5)
        if row == 0:
            centers = [sum(panel.get_position().intervalx) / 2 for panel in (ax, tx)]
            fig.legend([frontier_line, random_line],
                       ['Pareto frontier', 'Random mixture'], ncol=2,
                       loc='upper center', bbox_to_anchor=(centers[0], .987),
                       handlelength=2.1, handletextpad=.5, columnspacing=1.1)
            always_proxy = Line2D([], [], color='#b35c20', marker='s', ls='', mfc='white')
            fig.legend([latency_line, always_proxy], ['Pareto frontier', 'W: GPT-5.5'], ncol=2,
                       loc='upper center', bbox_to_anchor=(centers[1], .987),
                       handletextpad=.4, columnspacing=1.1)
            fig.text(centers[0], .927, 'Higher accuracy, fewer expert calls',
                     ha='center', fontsize=8, color=BLUE)
            fig.text(centers[1], .927, 'Higher accuracy, shorter waiting time',
                     ha='center', fontsize=8, color=BLUE)
        assert np.array_equal(gate_line.get_xdata(), x)
        assert np.array_equal(gate_line.get_ydata(), y)
        drawn[pool] = {'n': p['never']['n'], 'query_weighting': 'unit', 'accuracy_x': x.tolist(), 'accuracy_y': y.tolist(),
                       'random_x': [0, 100], 'random_y': [float(y[0]), float(y[-1])],
                       'always_reference': float(y[-1]),
                       'call_rate_pareto_arms': [ARMS[i] for i in frontier],
                       'latency_pareto_arms': [ARMS[i] for i in latency_frontier],
                       'metric_sources': METRIC_SOURCES,
                       'latency_comparison_x': latency.tolist(),
                       'timing': {key: [p[a][key] for a in ARMS]
                                  for key in ('mean_s', 'p50_s')}}

    check_figure(data, drawn)
    fig.text(.085, .031,
             'L: local; C/B/A: conservative/balanced/aggressive; W: always-escalate via GPT-5.5.',
             fontsize=6.7, va='bottom', style='italic')
    fig.text(.085, .011, 'Accuracy: judged benchmark. TTFA: full real-time timing run; early answer audio contributes zero wait.',
             fontsize=6.7, va='bottom', style='italic')
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / 'revision_accuracy_latency.pdf')
    fig.savefig(output_dir / 'revision_accuracy_latency.png', dpi=220)
    plt.close(fig)
    (output_dir / 'plotted_values.json').write_text(json.dumps(drawn, indent=2) + '\n')
    if output_dir.resolve() == (HERE / 'figures').resolve():
        (HERE / 'revision_data/academic_figure_values.json').write_text(json.dumps(drawn, indent=2) + '\n')
        mirror = HERE.parent / 'figures'
        mirror.mkdir(parents=True, exist_ok=True)
        for suffix in ('pdf', 'png'):
            name = 'revision_accuracy_latency.' + suffix
            shutil.copy2(output_dir / name, mirror / name)
    return drawn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-summary', type=Path, default=HERE / 'revision_data/native_summary.json')
    parser.add_argument('--internal-summary', '--reweighting', dest='internal_summary', type=Path, default=HERE / 'revision_data/internal_unweighted_summary.json')
    parser.add_argument('--ttfa-summary', type=Path, default=HERE.parent / 'ttfa_real/nonnegative/summary.json')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'figures')
    args = parser.parse_args()
    make_figure(args.native_summary, args.output_dir, args.internal_summary, args.ttfa_summary)
