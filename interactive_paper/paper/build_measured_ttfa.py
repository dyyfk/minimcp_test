"""Audit the recorded ttfa1 sessions and build the paper's timing table.

No inference or judging. All timing statistics use completed sessions;
failed sessions remain in the counts. The accuracy ledger stays separate.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import runpy
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = HERE.parent / 'ttfa_real'
ARMS = ['local', 'conservative', 'balanced', 'aggressive', 'always']
POOLS = ['frozen', 'striviaqa', 'swebq', 'sllama', 'sdqa']
NAMES = dict(zip(POOLS, ['Internal', 'Speech TriviaQA', 'Speech Web Questions',
                         'Speech Llama Questions', 'SD-QA']))
SOURCE_COMMIT = '9e86247'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display(value):
    return format(Decimal(str(value)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP), '.2f')


def build():
    spec = importlib.util.spec_from_file_location('ttfa_summary', SOURCE / 'summarize_ttfa.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with contextlib.redirect_stdout(io.StringIO()):
        check = runpy.run_path(str(SOURCE / 'verify_rows.py'))['check']
    files = sorted((SOURCE / 'results/ttfa1').glob('*/*.jsonl.shard*'))
    rows = [json.loads(line) for path in files for line in path.read_text().splitlines()
            if line.strip()]
    keys = [(r['pool'], r['arm'], r['id'], r['attempt']) for r in rows]
    assert len(keys) == len(set(keys)) == 5950
    assert {r['attempt'] for r in rows} == {1}
    assert {r['run_id'] for r in rows} == {'ttfa1'}
    assert {r['protocol'] for r in rows} == {'ttfa-v3'}
    frozen = json.loads((SOURCE / 'freeze.json').read_text())
    summary = json.loads((SOURCE / 'summary_ttfa1_final.json').read_text())
    assert not summary['order_violations']
    assert {r['gate_art_sha256'] for r in rows} == {frozen['gate_art_sha256']}
    assert all(not check(r) for r in rows)
    gate_info = {}
    for name in ['gate_native.json', 'gate_native_noleak.json']:
        raw = subprocess.check_output(['git', 'show',
            f'{SOURCE_COMMIT}:interactive_paper/data/{name}'], cwd=ROOT)
        gate = json.loads(raw)
        gate_info[name] = {'sha256': hashlib.sha256(raw).hexdigest(),
                           'train_n': gate['train_n'],
                           'thresholds_en': gate['eot_thresholds_lang']['en']}
    assert gate_info['gate_native.json']['sha256'] == frozen['gate_art_sha256']
    assert gate_info['gate_native_noleak.json']['sha256'] != frozen['gate_art_sha256']

    pools = {}
    timestamp_max_error = 0.
    for pool in POOLS:
        manifest = [json.loads(line) for line in
                    (SOURCE / f'{pool}_manifest.jsonl').read_text().splitlines()]
        ids = {r['id'] for r in manifest}
        assert len(ids) == frozen['pools'][pool]['n_queries']
        arms = {}
        valid_ids = []
        for arm in ARMS:
            rs = [r for r in rows if r['pool'] == pool and r['arm'] == arm]
            assert {r['id'] for r in rs} == ids
            assert len(rs) == len(ids)
            assert len({r['config_sha256'] for r in rs}) == 1
            if arm not in ('local', 'always'):
                assert {r['threshold'] for r in rs} == {
                    frozen['eot_thresholds_lang']['en'][arm]}
            ok = [r for r in rs if r['status'] == 'completed'
                  and r['ttfa_answer_s'] is not None]
            valid_ids.append({r['id'] for r in ok})
            for r in rs:
                ts = r['ts']
                if r['ttfa_answer_s'] is not None:
                    error = abs(ts['first_answer_pcm'] - ts['input_end'] - r['ttfa_answer_s'])
                    timestamp_max_error = max(timestamp_max_error, error)
                    assert error <= .00101  # timestamps and differences rounded independently
                if r['fired']:
                    assert r['expert_input_s'] <= min(r['input_s'], ts['fire']) + .002
                assert ts['first_answer_pcm'] == ts[
                    'relay_first_pcm' if r['fired'] else 'local_first_pcm']
            d = mod.dist([r['ttfa_answer_s'] for r in ok])
            old = summary['pools'][pool]['arms'][arm]
            assert d == old['ttfa_answer']
            assert dict(Counter(r['status'] for r in rs)) == old['status_counts']
            assert len(ok) == old['n_valid_ttfa']
            early = sum(r['ttfa_answer_s'] < 0 for r in ok)
            assert early == old['n_early_response']
            calls = sum(r['fired'] for r in rs)
            onset = sum(r['onset_chunk'] is not None for r in rs)
            assert round(calls / onset, 4) == old['escalation_rate']
            arms[arm] = {'n': len(rs), 'valid': len(ok), 'failed': len(rs)-len(ok),
                         'early': early, 'calls': calls, 'onset': onset,
                         'call_rate_all_sessions': calls / len(rs),
                         'ttfa_answer': d,
                         'hardware_counts': dict(sorted(Counter(r['gpu'] for r in rs).items()))}
        common = set.intersection(*valid_ids)
        paired = summary['pools'][pool]['paired']
        assert len(common) == paired['n_common_valid_ids']
        for arm in ARMS:
            ds = mod.dist([r['ttfa_answer_s'] for r in rows if r['pool'] == pool
                           and r['arm'] == arm and r['id'] in common])
            assert ds == paired[arm]
        pools[pool] = {'arms': arms, 'paired': paired}

    # Each query's actual input identity agrees across all five arms.
    for pool in POOLS:
        ids = {r['id'] for r in rows if r['pool'] == pool}
        for qid in ids:
            assert len({r['input_sha256'] for r in rows
                        if r['pool'] == pool and r['id'] == qid}) == 1
    audit = {
        'source_commit': SOURCE_COMMIT, 'run_id': 'ttfa1', 'protocol': 'ttfa-v3',
        'definition': 'first_answer_pcm - input_end; signed server PCM-ready seconds',
        'population': 'completed sessions with a non-null answer timestamp',
        'n_sessions': len(rows), 'status_counts': dict(sorted(Counter(r['status'] for r in rows).items())),
        'hardware_counts': dict(sorted(Counter(r['gpu'] for r in rows).items())),
        'timestamp_max_rounding_error_s': round(timestamp_max_error, 6),
        'duplicate_rows': 0, 'row_check_failures': 0,
        'summary_statistics_match': True, 'gate_artifacts': gate_info,
        'scope': 'Separate unjudged timing sessions; gate differs from the overlap-remediated accuracy fit. No joint accuracy-latency measurement.',
        'source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [
            SOURCE / 'freeze.json', SOURCE / 'summary_ttfa1_final.json',
            SOURCE / 'summarize_ttfa.py', SOURCE / 'verify_rows.py',
            *sorted(SOURCE.glob('*_manifest.jsonl')), *files]},
        'pools': pools,
    }
    (HERE / 'revision_data/measured_ttfa.json').write_text(json.dumps(audit, indent=2)+'\n')
    table = [r'\begin{table}[!ht]', r'\centering\small',
             r'\caption{Measured server PCM-ready TTFA (seconds). Each arm attempts every query in the pool. Statistics use completed sessions; Fail counts all other outcomes, and Early counts negative TTFA among completed sessions. Calls uses all attempted sessions as its denominator. One run per query and arm; quantiles describe query variation.}',
             r'\label{tab:latency}', r'\setlength{\tabcolsep}{5pt}',
             r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
             r'arm & Valid & Fail & Early & Calls (\%) & Mean & P50 & P95 & P99 \\', r'\midrule']
    for i, pool in enumerate(POOLS):
        if i:
            table.append(r'\midrule')
        table.append(r'\multicolumn{9}{l}{\textit{' + NAMES[pool] +
                     f" ($n={pools[pool]['arms']['local']['n']}$ per arm)" + r'}} \\')
        for arm in ARMS:
            a = pools[pool]['arms'][arm]
            d = a['ttfa_answer']
            cells = [arm, str(a['valid']), str(a['failed']), str(a['early']),
                     f"{100*a['call_rate_all_sessions']:.1f}",
                     *[display(d[k]) for k in ['mean', 'p50', 'p95', 'p99']]]
            table.append(' & '.join(cells) + r' \\')
    table.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}', ''])
    (HERE / 'sections/measured_ttfa_table.tex').write_text('\n'.join(table))
    print(json.dumps({k:audit[k] for k in ['n_sessions','status_counts','hardware_counts',
                     'timestamp_max_rounding_error_s','summary_statistics_match']}, indent=2))
    return audit


if __name__ == '__main__':
    build()
