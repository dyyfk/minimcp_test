"""Regenerate benchmark figures and appendix rates from native benchmark rows.

The original teaser_v2 figure and original main Table 1 are preserved unchanged.
This script never rewrites sections/revision_table.tex.

Usage: python build_revision_figures.py --data-dir ../data/native_bench
Requires numpy, pandas, pyarrow, matplotlib. Does not run models or call APIs.
Input provenance: minimcp_test e74ca0c80f0e20fea8b466424e0307ea621a1403.
Timing intentionally reproduces the prior plotting recipe as a diagnostic;
it must not be interpreted as comparable client-side audible completion.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ARMS = ['never', 'conservative', 'balanced', 'aggressive', 'always']
SPECS = {
    'frozen': ('Internal', 'adequate'),
    'striviaqa': ('Speech TriviaQA', 'oab_ok'),
    'swebq': ('Speech WebQ', 'oab_ok'),
    'sllama': ('Speech Llama Q.', 'oab_ok'),
    'sdqa': ('SD-QA', 'adequate'),
    'sreason': ('Reasoning QA (zh)', 'adequate'),
    'valpaca': ('AlpacaEval', 'vb_score'),
}
BLUE, ORANGE, GREY, GREEN = '#2466A3', '#CB6B25', '#67727D', '#267968'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})


def load_rows(data_dir, pool, arm, col):
    name = f'{pool}_{arm}{"" if arm == "never" else "_tts"}_judged.parquet'
    path = data_dir / name
    df = pd.read_parquet(path).drop_duplicates('id', keep='last')
    df = df[df[col].notna()].set_index('id').copy()
    for key in ['n_chunks', 'onset_chunk', 'relay_synth_ms', 'relay_audio_s', 'relay_ms', 'stall_ms', 'wait_chunks', 'answer_ms']:
        if key in df:
            df[key] = pd.to_numeric(df[key], errors='raise').astype(float)
    esc = df['mode'].eq('escalated')
    on = (df['onset_chunk'].fillna(df['n_chunks']) + 1 - df['n_chunks']).clip(lower=0)
    # Time to first audio (TTFA). Escalated: the relay waveform is ready to
    # play once the blocking TTS call returns (relay_synth_ms); the waveform's
    # own duration (relay_audio_s) is playback, not latency, and is reported
    # separately. Local: the talker's answer text is complete (answer_ms is
    # timed from onset), an upper bound on its first audio.
    synth = df['relay_synth_ms'].fillna(0) / 1000 if 'relay_synth_ms' in df else df['relay_ms'].fillna(0) / 1000
    df['timing_diagnostic_s'] = np.where(esc, on + df['stall_ms'].fillna(0) / 1000 + df['wait_chunks'].fillna(0) + synth, on + df['answer_ms'].fillna(0) / 1000)
    assert not df['timing_diagnostic_s'].isna().any(), name
    v = df['timing_diagnostic_s']
    stats = {'n': len(df), 'accuracy': float(df[col].astype(float).mean()),
             'rate': float(esc.mean()), 'mean_s': float(v.mean()),
             'p50_s': float(v.quantile(.5)), 'p95_s': float(v.quantile(.95)),
             'p99_s': float(v.quantile(.99)),
             'relay_playback_mean_s': float(df.loc[esc, 'relay_audio_s'].fillna(0).mean()) if esc.any() and 'relay_audio_s' in df else 0.0,
             'early_onset_n': int((df['onset_chunk'] < df['n_chunks'] - 1).sum()),
             'early_escalation_n': int(((df['onset_chunk'] < df['n_chunks'] - 1) & esc).sum()),
             'source': 'interactive_paper/data/native_bench/' + name,
             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    ecol = col + '_expert'
    if arm == 'always' and ecol in df:
        matched = df[ecol].notna()
        stats['expert_scored_n'] = int(matched.sum())
        stats['expert_text_accuracy'] = float(df.loc[matched, ecol].astype(float).mean())
        stats['paired_relay_accuracy'] = float(df.loc[matched, col].astype(float).mean())
    return df, stats


def generate(data_dir):
    out = HERE / 'revision_data'; out.mkdir(exist_ok=True)
    summary = {}
    for pool, (_, col) in SPECS.items():
        summary[pool] = {}
        ids = None
        for arm in ARMS:
            df, stats = load_rows(data_dir, pool, arm, col)
            if ids is None: ids = set(df.index)
            assert set(df.index) == ids, f'{pool}: arm ID mismatch'
            summary[pool][arm] = stats
        p = summary[pool]
        for arm in ARMS:
            # Preserve Table 1's published policy-endpoint mixture convention.
            mix_weight = p[arm]['rate']
            ref = p['never']['accuracy'] + mix_weight * (p['always']['accuracy'] - p['never']['accuracy'])
            p[arm]['matched_mixture_accuracy'] = ref
            p[arm]['gain_vs_mixture'] = p[arm]['accuracy'] - ref
            exact_ref = p['never']['accuracy'] + mix_weight / p['always']['rate'] * (p['always']['accuracy'] - p['never']['accuracy'])
            p[arm]['actual_rate_mixture_accuracy'] = exact_ref
            p[arm]['gain_vs_actual_rate_mixture'] = p[arm]['accuracy'] - exact_ref
    doc = {'source_commit': 'e74ca0c80f0e20fea8b466424e0307ea621a1403',
           'runs_per_query_arm': 1,
           'random_reference': 'original Table 1: (1-r)*local_accuracy + r*always_accuracy, where r is the gate rate; always-policy endpoint treated as 100%',
           'sensitivity_reference': 'actual-rate normalization uses lambda=r/realized_always_rate; retained separately',
           'timing_status': 'reconstructed time to first audio (TTFA): escalated = onset + stall + wait_chunks + relay TTS synthesis; local = onset + answer text complete. Relay playback duration (relay_audio_s) is excluded and reported as relay_playback_mean_s',
           'pools': summary}
    (out / 'native_summary.json').write_text(json.dumps(doc, indent=2) + '\n')
    write_rate_table(summary)
    draw_pairs(summary)
    print(json.dumps({p: {a: {k: v for k, v in x.items() if k in ['rate', 'mean_s','p95_s','gain_vs_mixture']} for a,x in q.items()} for p,q in summary.items() if p in ['frozen','striviaqa','sdqa']}, indent=2))


def write_rate_table(s):
    lines = [r'\begin{table}[h]', r'\centering\small',
        r'\caption{Realized escalation rates (\%) for the MiniCPM native results in the upper block and caption of Table~\ref{tab:transfer}. These are measured rates, distinct from the nominal calibration budgets and from the NVDA replay budgets. Local-only makes no expert calls.}',
        r'\label{tab:native-rates}', r'\begin{tabular}{lrrrrr}', r'\toprule',
        r'Pool & $n$ & Conservative & Balanced & Aggressive & Always \\', r'\midrule']
    for pool in SPECS:
        q=s[pool]
        row=[SPECS[pool][0],str(q['never']['n'])]+[f"{100*q[a]['rate']:.1f}" for a in ARMS[1:]]
        lines.append(' & '.join(row)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (HERE/'sections/native_rates.tex').write_text('\n'.join(lines)+'\n')


def draw_pairs(s):
    fig, axes=plt.subplots(3,2,figsize=(7.1,6.35),gridspec_kw={'hspace':.42,'wspace':.31})
    for i,pool in enumerate(['frozen','striviaqa','sdqa']):
        p=s[pool];ax,tx=axes[i]
        x=np.array([p[a]['rate']*100 for a in ARMS]);y=np.array([p[a]['accuracy']*100 for a in ARMS])
        ax.grid(axis='y',color='#E5E9ED',lw=.65,zorder=0)
        ax.plot([0,100],[y[0],y[-1]],'--',lw=1.2,color=GREY,label='Random mixture')
        ax.axhline(y[-1],ls=':',lw=1.25,color=GREEN)
        ax.plot(x,y,'o-',color=BLUE,lw=1.8,ms=4.7,zorder=4)
        ax.scatter([x[-1]],[y[-1]],color=GREEN,marker='s',s=28,zorder=5)
        for j,label in enumerate(['L','C','B','A','W']):
            dx,dy = (4,-14) if j in [1,2,3] else ((3,6) if j==0 else (-15,-15))
            ax.annotate(label,(x[j],y[j]),xytext=(dx,dy),textcoords='offset points',fontsize=8,color=BLUE if j<4 else GREEN)
        ax.text(3,y[-1]+3,f"Always reference: {y[-1]:.1f}%",va='bottom',color=GREEN,fontsize=8)
        ax.set_ylim(30,104);ax.set_xlim(-3,105);ax.set_yticks([40,60,80,100]);ax.set_xticks([0,25,50,75,100]);ax.set_ylabel('Accuracy (%)')
        ax.set_title(f"{SPECS[pool][0]}  (n={p['never']['n']})",loc='left',fontweight='bold',fontsize=9.2,pad=5)
        tx.grid(axis='y',color='#E5E9ED',lw=.65,zorder=0)
        for key,lab,color,ls,marker in [('mean_s','Mean',BLUE,'-','o'),('p50_s','P50',GREY,'--','s'),('p95_s','P95',ORANGE,':','^')]:
            tx.plot(range(5),[p[a][key] for a in ARMS],label=lab,color=color,ls=ls,lw=1.4,marker=marker,ms=4)
        tx.set_yscale('log');tx.set_ylim(1,350);tx.set_yticks([1,3,10,30,100,300]);tx.set_yticklabels(['1','3','10','30','100','300']);tx.minorticks_off()
        tx.set_xticks(range(5));tx.set_xticklabels(['L','C','B','A','W']);tx.set_xlim(-.15,4.15);tx.set_ylabel('Time to first audio (s)')
        tx.set_title('Time to first audio',loc='left',fontsize=9.2,pad=5)
        if i==0: tx.legend(frameon=False,fontsize=7.5,loc='upper left',ncol=3,handlelength=1.2,columnspacing=.7)
        if i==2: ax.set_xlabel('Realized expert call rate (%)');tx.set_xlabel('Recorded arm')
    fig.subplots_adjust(left=.09,right=.985,top=.95,bottom=.075)
    fig.savefig(HERE/'figures/revision_accuracy_latency.pdf')
    fig.savefig(HERE/'figures/revision_accuracy_latency.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=HERE.parent/'data/native_bench')
    generate(parser.parse_args().data_dir)
