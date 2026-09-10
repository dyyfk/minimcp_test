"""Restore reporting on all 240 Internal queries with unit query weights.

Reaggregate archived MiniCPM outcomes and independent TTFA; retain the
original NVDA pass-3 replay policies. No inference, judging, or refitting.
"""
import argparse
from collections import Counter
import copy
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent
ARMS = ['never', 'conservative', 'balanced', 'aggressive', 'always']
CATEGORIES = {'easy-chat': 'Chat', 'easy-fact': 'Factual QA',
              'hard-knowledge': 'Knowledge', 'hard-math': 'Math', 'trap': 'SimpleQA traps'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_json(name):
    path = 'interactive_paper/data/gate_pull/new/probe_doc/' + name
    content = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=REPO)
    return json.loads(content), {'git_path': path, 'sha256': hashlib.sha256(content).hexdigest()}


def build(args):
    old = json.loads((HERE/'revision_data/native_summary.json').read_text())['pools']['frozen']
    diag = json.loads((HERE/'revision_data/internal_diagnostic.json').read_text())
    assert digest(args.queries) == diag['input_sha256']['internal_query_manifest.jsonl']
    meta = pd.read_json(args.queries, lines=True).set_index('id')
    assert meta.index.is_unique
    ids = sorted(meta.index[meta.split.eq('test')].tolist())
    assert len(ids) == 240
    counts = meta.loc[ids, 'pool'].value_counts().to_dict()
    assert counts == {'easy-chat': 60, 'easy-fact': 40, 'hard-knowledge': 60, 'hard-math': 60, 'trap': 20}
    hashes = {str(args.queries): digest(args.queries)}
    frames, minicpm = {}, {}
    for arm in ARMS:
        path = args.data_dir / Path(old[arm]['source']).name
        assert digest(path) == old[arm]['sha256']
        hashes[str(path)] = digest(path)
        frame = pd.read_parquet(path).set_index('id')
        assert frame.index.is_unique and set(frame.index) == set(ids)
        frame = frame.loc[ids]
        assert frame['query'].eq(meta.loc[ids, 'query']).all()
        assert frame.adequate.isin([True, False]).all()
        accuracy, rate = float(frame.adequate.mean()), float(frame['mode'].eq('escalated').mean())
        assert abs(accuracy-old[arm]['accuracy']) < 1e-12
        assert abs(rate-old[arm]['rate']) < 1e-12
        minicpm[arm] = {'n': 240, 'correct': int(frame.adequate.sum()), 'accuracy': accuracy,
            'n_escalated': int(frame['mode'].eq('escalated').sum()), 'rate': rate,
            'source': old[arm]['source'], 'sha256': digest(path), 'source_n': 240}
        frames[arm] = frame
    local, expert = (minicpm[a]['accuracy'] for a in ['never', 'always'])
    for arm, row in minicpm.items():
        row['matched_mixture_accuracy'] = (1-row['rate'])*local+row['rate']*expert
        assert abs(row['matched_mixture_accuracy']-old[arm]['matched_mixture_accuracy']) < 1e-12
    assert [minicpm[a]['correct'] for a in ARMS] == [98, 112, 130, 151, 169]
    outcomes = np.column_stack([frames[a].adequate.to_numpy(float) for a in ARMS])
    rng = np.random.default_rng(42)
    boot = np.empty((100000, 5))
    for start in range(0, 100000, 2000):
        indices = rng.integers(0, 240, size=(2000, 240))
        boot[start:start+2000] = outcomes[indices].mean(axis=1)
    ci = np.quantile(boot, [.025, .975], axis=0)
    for j, arm in enumerate(ARMS): minicpm[arm]['accuracy_ci95'] = ci[:, j].tolist()
    strata = []
    for pool, label in CATEGORIES.items():
        subset = meta.loc[ids].index[meta.loc[ids, 'pool'].eq(pool)]
        strata.append({'category': pool, 'label': label, 'n': len(subset),
            **{a: float(frames[a].loc[subset, 'adequate'].mean()) for a in ARMS},
            'aggressive_rate': float(frames['aggressive'].loc[subset, 'mode'].eq('escalated').mean()),
            'audio_gt_30s_n': int(frames['always'].loc[subset, 'audio_s'].gt(30).sum())})

    judged, judged_source = git_json('internal_pass3_judged.json')
    replay, replay_source = git_json('internal_pass3_remix.json')
    rows = pd.DataFrame(judged['rows']).set_index('id')
    assert rows.index.is_unique and set(rows.index) == set(ids)
    rows = rows.loc[ids]
    assert rows['query'].eq(meta.loc[ids, 'query']).all()
    assert rows.adequate.isin([True, False]).all()
    experts = pd.read_parquet(args.nvda_expert).set_index('id')
    assert experts.index.is_unique and experts.loc[ids, 'expert_adequate'].isin([True, False]).all()
    rows['expert'] = experts.loc[ids, 'expert_adequate']
    rows['pool'] = meta.loc[ids, 'pool']
    scoreable = rows.onset_frame.ge(0)
    assert int(scoreable.sum()) == 223
    assert abs(rows.adequate.mean()-replay['rows']['always-local']['accuracy']) < 1e-12
    assert abs(rows.expert.mean()-replay['rows']['always-expert']['accuracy']) < 1e-12
    for pool, group in rows.groupby('pool'):
        prior = replay['by_pool'][pool]
        assert len(group) == prior['n']
        assert abs(group.adequate.mean()-prior['local']) < 1e-12
        assert abs(group.expert.mean()-prior['expert']) < 1e-12
    nvda = {}
    for arm, key in zip(ARMS, ['always-local', 'gate@0.15', 'gate@0.3', 'gate@0.5', 'always-expert']):
        source = replay['rows'][key]
        nvda[arm] = {'n': 240, 'accuracy': source['accuracy'], 'rate': source['call_rate'],
                     'correct': round(source['accuracy']*240)}
        if arm in ARMS[1:4]:
            k = source['n_escalated']
            random = rows.adequate.mean() + (k/223)*(rows.loc[scoreable, 'expert'].astype(float)-rows.loc[scoreable, 'adequate'].astype(float)).sum()/240
            assert abs(random-source['matched_random']) < 1e-12
            nvda[arm].update(n_escalated=k, matched_mixture_accuracy=float(random))
    hashes[str(args.nvda_expert)] = digest(args.nvda_expert)

    timing_source = HERE.parent/'ttfa_real/nonnegative/summary.json'
    old_timing = json.loads(timing_source.read_text())
    assert old_timing['formula'] == 'max(0, ts.first_answer_pcm - ts.input_end)'
    timing = {'formula': old_timing['formula'], 'n': 240, 'arms': {}}
    expected_hash = {s['path']: s['sha256'] for s in old_timing['source_files']}
    for arm in ['local', *ARMS[1:]]:
        entries = []
        for path in sorted((args.ttfa_results/'frozen').glob(arm+'.jsonl.shard*')):
            assert digest(path) == expected_hash[str(path.relative_to(args.ttfa_results))]
            hashes[str(path)] = digest(path)
            entries.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
        assert len(entries) == 240 and {e['id'] for e in entries} == set(ids)
        waits, early = [], 0
        for entry in entries:
            assert (entry['pool'],entry['arm'],entry['protocol']) == ('frozen',arm,'ttfa-v3')
            if entry['status'] != 'completed': continue
            ts = entry['ts']; first, end = ts['first_answer_pcm'], ts['input_end']
            assert first == (ts['relay_first_pcm'] if entry['fired'] else ts['local_first_pcm'])
            signed = round(first-end, 6)
            assert abs(signed-entry['ttfa_answer_s']) <= .001001
            waits.append(max(0.,signed)); early += signed < 0
        prior = old_timing['pools']['frozen']['arms'][arm]
        assert len(waits) == prior['n_completed'] and early == prior['n_early_response']
        assert dict(Counter(e['status'] for e in entries)) == prior['status_counts']
        for name, actual in {'mean': np.mean(waits), **{f'p{q}':np.percentile(waits,q) for q in [50,95,99]}}.items():
            assert abs(actual-prior['nonnegative_ttfa_s'][name]) < 1e-12
        timing['arms'][arm] = copy.deepcopy(prior)
    hashes[str(timing_source)] = digest(timing_source)
    return {'n': 240, 'source_n': 240, 'category_counts': counts,
        'category_weights': {pool:1.0 for pool in counts}, 'sum_weights': 240,
        'selection_context': 'All original 240 Internal test IDs retained; each query has unit weight.',
        'included_ids': ids, 'excluded_ids': [],
        'minicpm': {'arms': minicpm, 'strata': strata,
            'bootstrap': {'replicates':100000, 'seed':42, 'method':'Paired query percentile bootstrap over all 240 IDs; fixed recorded arms'},
            'derived': {'aggressive_gain_over_local_pp':100*(minicpm['aggressive']['accuracy']-local),
                'aggressive_gain_over_random_pp':100*(minicpm['aggressive']['accuracy']-minicpm['aggressive']['matched_mixture_accuracy']),
                'local_to_always_gap_recovered_pct':100*(minicpm['aggressive']['accuracy']-local)/(expert-local)}},
        'nvda': {'arms':nvda, 'n_scoreable':223, 'n_no_onset':17,
            'selective_accuracy_available':True,
            'verification':'Archived full-cohort selective accuracies retained; local/expert outcomes and scoreability-matched random references recomputed. Probe ranking was not rerun.',
            'limitation':'Architecture selected using the original Internal split; post-selection diagnostic.',
            'judged_source':judged_source, 'original_replay_source':replay_source},
        'timing':timing, 'source_sha256':hashes}


def format_number(value, places=1):
    quantum = Decimal(1).scaleb(-places)
    return format(Decimal(str(round(value, 10))).quantize(quantum, rounding=ROUND_HALF_UP), f'.{places}f')


def table_text(report):
    arms = report['minicpm']['arms']
    labels = ['local-only', 'conservative', 'balanced', 'aggressive', 'always']
    lines = [r'\begin{table}[h]', r'\centering\small',
        r'\caption{Internal results on all 240 original test queries (\%), with unit query weights. Random mixes the measured local/always endpoints at the corresponding call rate. Accuracy has pointwise 95\% percentile intervals from 100,000 paired query-bootstrap resamples (seed 42), conditional on the fixed test set and recorded arms. The intervals exclude judging and model-rerun uncertainty.}',
        r'\label{tab:live}', r'\begin{tabular}{lrrr}', r'\toprule',
        r'tier & calls & accuracy [95\% CI] & random \\', r'\midrule']
    for arm, label in zip(ARMS, labels):
        row = arms[arm]
        ci = ','.join(format_number(v*100) for v in row['accuracy_ci95'])
        cells = [label, format_number(row['rate']*100), format_number(row['accuracy']*100)+f' [{ci}]',
                 format_number(row['matched_mixture_accuracy']*100) if arm in ARMS[1:4] else '---']
        lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    result = {'internal_unweighted_results.tex':'\n'.join(lines)+'\n'}
    timing = report['timing']['arms']
    t_arms = ['local', *ARMS[1:]]
    completed = '/'.join(str(timing[a]['n_completed']) for a in t_arms)
    failed = '/'.join(str(timing[a]['n_failed']) for a in t_arms)
    early = '/'.join(str(timing[a]['n_early_response']) for a in t_arms)
    caption = (r'\caption{Internal server TTFA (seconds), independent \texttt{ttfa-v3} run on all 240 test IDs: '
        f'240 attempts per arm, with {completed} completed and {failed} failed in row order. '
        'TTFA is the nonnegative wait from scheduled input end to first answer PCM. All completed sessions '
        f'count; the {early} early responses have zero wait. Failed attempts have no TTFA value. '
        'Timing ends at the first local PCM-bearing chunk return or relay PCM yield, excluding '
        'candidate/stall audio and client playback.}')
    lines = [r'\begin{table}[!ht]',r'\centering\small',caption,r'\label{tab:latency}',
        r'\begin{tabular}{lrrrr}',r'\toprule',r'tier & mean & P50 & P95 & P99 \\',r'\midrule']
    for arm,label in zip(t_arms,labels):
        values = timing[arm]['nonnegative_ttfa_s']
        lines.append(' & '.join([label,*[format_number(values[k],2) for k in ['mean','p50','p95','p99']]])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    result['internal_latency.tex'] = '\n'.join(lines)+'\n'
    lines = [r'\begin{table}[h]',r'\centering\small',
        r"\caption{Descriptive strata of the full 240-query Internal test set, with unit query weights. Local, aggressive (A), and always report answer-content accuracy (\%); call rate is the aggressive arm's realized rate. The final column counts question waveforms longer than the expert's 30-second input window. All five categories and all original queries are retained.}",
        r'\label{tab:internal-strata}',r'\begin{tabular}{lrrrrrr}',r'\toprule',
        r'Stratum & $n$ & Local & A & Always & A call rate & $>30$\,s ($n$) \\',r'\midrule']
    for row in report['minicpm']['strata']:
        cells = [row['label'],str(row['n']),*[format_number(row[k]*100) for k in ['never','aggressive','always','aggressive_rate']],str(row['audio_gt_30s_n'])]
        lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    result['internal_strata.tex'] = '\n'.join(lines)+'\n'
    return result


def write_tables(report):
    for name,text in table_text(report).items(): (HERE/'sections'/name).write_text(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=WORKSPACE/'review_sources/native_bench')
    parser.add_argument('--queries', type=Path, default=WORKSPACE/'review_sources/internal_query_manifest.jsonl')
    parser.add_argument('--nvda-expert', type=Path, default=WORKSPACE/'tmp/nvda-internal-expert.parquet')
    parser.add_argument('--ttfa-results', type=Path, default=HERE.parent/'ttfa_real/results/ttfa1')
    parser.add_argument('--output', type=Path, default=HERE/'revision_data/internal_unweighted_summary.json')
    args = parser.parse_args()
    report = build(args)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    write_tables(report)
    print(json.dumps({'n':report['n'], 'minicpm':report['minicpm']['arms'],
                      'derived':report['minicpm']['derived'], 'nvda':report['nvda']['arms']},indent=2))
