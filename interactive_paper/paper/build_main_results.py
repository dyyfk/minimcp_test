"""Build and check both main-table blocks from the shared reporting sources.

Run this before build_academic_revision_figure.py. Use --check to detect stale
table cells without changing files. Internal uses the full 240 original IDs
with unit weights for both models; external pools retain their original results.
NVDA replay results remain a separate block.
"""
import argparse
import copy
from decimal import Decimal, ROUND_HALF_UP
import json
import subprocess
from math import fsum
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTERNAL_POOLS = ['striviaqa', 'swebq', 'sllama', 'sdqa']
TABLE_POOLS = ['frozen', *EXTERNAL_POOLS]
ROWS = [
    ('always-local', 'never', 'accuracy'),
    (r'\;+ gate, conservative', 'conservative', 'accuracy'),
    (r'\;+ gate, balanced', 'balanced', 'accuracy'),
    (r'\;+ gate, aggressive', 'aggressive', 'accuracy'),
    (r'\;matched random, aggressive', 'aggressive', 'matched_mixture_accuracy'),
    ('always-escalate', 'always', 'accuracy'),
]
START = '% BEGIN GENERATED MINICPM ROWS\n'
END = '% END GENERATED MINICPM ROWS'
NVDA_START = '% BEGIN GENERATED NVDA ROWS\n'
NVDA_END = '% END GENERATED NVDA ROWS'


def load_result_pools(native_path=HERE / 'revision_data/native_summary.json',
                      internal_path=HERE / 'revision_data/internal_unweighted_summary.json'):
    original = json.loads(native_path.read_text())['pools']
    report = json.loads(internal_path.read_text())
    pools = copy.deepcopy(original)
    assert report['n'] == len(report['included_ids']) == 240
    assert len(set(report['included_ids'])) == 240
    assert report['excluded_ids'] == []
    assert set(report['category_weights'].values()) == {1.0}
    for arm, result in report['minicpm']['arms'].items():
        source = original['frozen'][arm]
        if result['sha256'] != source['sha256'] or result['source_n'] != source['n']:
            raise ValueError(f'{arm}: Internal and original summaries use different source runs')
        for key in ('accuracy', 'rate', 'matched_mixture_accuracy', 'n'):
            pools['frozen'][arm][key] = result[key]
        expected = ((1-result['rate'])*report['minicpm']['arms']['never']['accuracy']
                    + result['rate']*report['minicpm']['arms']['always']['accuracy'])
        if abs(result['matched_mixture_accuracy']-expected) > 1e-12:
            raise ValueError(f'{arm}: random reference uses inconsistent included IDs')
    return pools


def load_figure_pools(native_path=HERE / 'revision_data/native_summary.json',
                      internal_path=HERE / 'revision_data/internal_unweighted_summary.json',
                      ttfa_path=HERE.parent / 'ttfa_real/nonnegative/summary.json'):
    pools = load_result_pools(native_path, internal_path)
    ttfa = json.loads(ttfa_path.read_text())
    internal_timing = json.loads(internal_path.read_text())['timing']
    assert ttfa['formula'] == 'max(0, ts.first_answer_pcm - ts.input_end)'
    for pool in ('frozen', 'striviaqa', 'sdqa'):
        for arm, result in pools[pool].items():
            if arm not in ('never', 'conservative', 'balanced', 'aggressive', 'always'):
                continue
            timing = internal_timing if pool == 'frozen' else ttfa['pools'][pool]
            measured = timing['arms']['local' if arm == 'never' else arm]
            assert measured['n_attempts'] == result['n']
            for key in ('mean', 'p50'):
                value = measured['nonnegative_ttfa_s'][key]
                assert value >= 0, (pool, arm, key)
                result[key + '_s'] = value
    return pools


def display(value, signed=False):
    # Remove binary-float noise before applying conventional one-decimal rounding.
    rounded = Decimal(str(round(value, 10))).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
    return format(rounded, '+.1f' if signed else '.1f')


def native_rows(pools):
    baseline = fsum(pools[p]['never']['accuracy'] for p in EXTERNAL_POOLS) / 4 * 100
    rows = []
    for label, arm, field in ROWS:
        values = [pools[p][arm][field] * 100 for p in TABLE_POOLS]
        macro = fsum(values[1:]) / len(EXTERNAL_POOLS)
        delta = macro - baseline
        cells = [display(v) for v in [*values, macro]]
        cells.append('---' if arm == 'never' else display(delta, signed=True))
        if arm == 'aggressive' and field == 'accuracy':
            label = r'\;\textbf{+ gate, aggressive}'
            cells = [r'\textbf{' + c + '}' for c in cells]
        rows.append({'label': label, 'arm': arm, 'field': field,
                     'values_pct': values, 'macro_pct': macro, 'gain_pp': delta,
                     'cells': cells})
    return rows


def rendered_rows(pools):
    return '\n'.join(' & '.join([row['label'], *row['cells']]) + r' \\'
                     for row in native_rows(pools)) + '\n'


def rendered_nvda_rows(internal_path=HERE / 'revision_data/internal_unweighted_summary.json'):
    internal = json.loads(internal_path.read_text())['nvda']['arms']
    path = 'interactive_paper/data/gate_pull/new/probe_doc/remix_eval3.json'
    archived = json.loads(subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=HERE.parents[1]))
    external = archived['v2: layer-avg x3, commit|onset_last|onset_mean8|run_mean']['loc_official']
    keys = ['local', 'gate@0.15', 'gate@0.3', 'gate@0.5', 'random@0.5', 'expert']
    baseline = fsum(external[p]['local'] for p in EXTERNAL_POOLS)/4*100
    lines = []
    for (label, arm, field), key in zip(ROWS, keys):
        value = internal[arm][field]
        values = [external[p][key]*100 for p in EXTERNAL_POOLS]
        average = fsum(values)/4
        cells = ['---' if value is None else display(value*100),
                 *[display(v) for v in values], display(average),
                 '---' if arm == 'never' else display(average-baseline, signed=True)]
        lines.append(' & '.join([label, *cells]) + r' \\')
    return '\n'.join(lines) + '\n'


def check_table(pools, table_path):
    text = table_path.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError('Main table must contain exactly one generated MiniCPM block')
    actual = text.split(START, 1)[1].split(END, 1)[0]
    if actual != rendered_rows(pools):
        raise ValueError('Main-table cells differ from reporting summaries; run build_main_results.py')
    if text.count(NVDA_START) != 1 or text.count(NVDA_END) != 1:
        raise ValueError('Main table must contain exactly one generated NVDA block')
    nvda = text.split(NVDA_START, 1)[1].split(NVDA_END, 1)[0]
    if nvda != rendered_nvda_rows():
        raise ValueError('NVDA table cells differ from reporting sources')
    return native_rows(pools)


def write_table(pools, table_path):
    text = table_path.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError('Main table must contain exactly one generated MiniCPM block')
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    table_path.write_text(before + START + rendered_rows(pools) + END + after)
    text = table_path.read_text()
    before, rest = text.split(NVDA_START, 1)
    _, after = rest.split(NVDA_END, 1)
    table_path.write_text(before + NVDA_START + rendered_nvda_rows() + NVDA_END + after)


def check_figure(pools, drawn):
    for pool, plot in drawn.items():
        arms = ['never', 'conservative', 'balanced', 'aggressive', 'always']
        expected_y = [pools[pool][arm]['accuracy'] * 100 for arm in arms]
        expected_x = [pools[pool][arm]['rate'] * 100 for arm in arms]
        if plot['accuracy_y'] != expected_y or plot['accuracy_x'] != expected_x:
            raise ValueError(f'{pool}: plotted accuracy/rate differs from the main-table source')
        if plot['random_y'] != [expected_y[0], expected_y[-1]]:
            raise ValueError(f'{pool}: random-mixture endpoints differ')
        if plot['always_reference'] != expected_y[-1]:
            raise ValueError(f'{pool}: always reference differs')
        for key, values in plot['timing'].items():
            if values != [pools[pool][arm][key] for arm in arms]:
                raise ValueError(f'{pool}: timing differs for {key}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--native-summary', type=Path, default=HERE / 'revision_data/native_summary.json')
    parser.add_argument('--internal-summary', type=Path, default=HERE / 'revision_data/internal_unweighted_summary.json')
    parser.add_argument('--table', type=Path, default=HERE / 'sections/revision_table.tex')
    parser.add_argument('--figure-values', type=Path)
    parser.add_argument('--ttfa-summary', type=Path, default=HERE.parent / 'ttfa_real/nonnegative/summary.json')
    args = parser.parse_args()
    pools = load_result_pools(args.native_summary, args.internal_summary)
    if not args.check:
        write_table(pools, args.table)
    rows = check_table(pools, args.table)
    if args.figure_values:
        check_figure(load_figure_pools(args.native_summary, args.internal_summary, args.ttfa_summary),
                     json.loads(args.figure_values.read_text()))
    print(json.dumps({'external_pools': EXTERNAL_POOLS,
                      'native_table_rows_checked': len(rows),
                      'figure_checked': args.figure_values is not None,
                      'macro_accuracy_pct': {r['label']: display(r['macro_pct']) for r in rows}}, indent=2))
