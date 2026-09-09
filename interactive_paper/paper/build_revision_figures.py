"""Regenerate benchmark figures and appendix rates from native benchmark rows.

The MiniCPM main-table rows and Figure 3 use this same summary.
The separate NVDA replay block and original teaser_v2 remain unchanged.

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

import numpy as np
import pandas as pd
from build_main_results import write_table
from build_academic_revision_figure import make_figure

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
    write_table(summary, HERE / 'sections/revision_table.tex')
    write_rate_table(summary)
    make_figure(out / 'native_summary.json', HERE / 'figures')
    print(json.dumps({p: {a: {k: v for k, v in x.items() if k in ['rate', 'mean_s','p95_s','gain_vs_mixture']} for a,x in q.items()} for p,q in summary.items() if p in ['frozen','striviaqa','sdqa']}, indent=2))


def write_rate_table(s):
    lines = [r'\begin{table}[h]', r'\centering\small',
        r'\caption{Realized escalation rates (\%) for the MiniCPM native results in the upper block of Table~\ref{tab:transfer}, plus the archived Mandarin and separate AlpacaEval pools. These are measured rates, distinct from the nominal calibration budgets and from the NVDA replay budgets. Local-only makes no expert calls.}',
        r'\label{tab:native-rates}', r'\begin{tabular}{lrrrrr}', r'\toprule',
        r'Pool & $n$ & Conservative & Balanced & Aggressive & Always \\', r'\midrule']
    for pool in SPECS:
        q=s[pool]
        row=[SPECS[pool][0],str(q['never']['n'])]+[f"{100*q[a]['rate']:.1f}" for a in ARMS[1:]]
        lines.append(' & '.join(row)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (HERE/'sections/native_rates.tex').write_text('\n'.join(lines)+'\n')



if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=HERE.parent/'data/native_bench')
    generate(parser.parse_args().data_dir)
