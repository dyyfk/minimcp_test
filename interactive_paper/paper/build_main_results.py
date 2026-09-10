"""Build and check the MiniCPM main-table rows from the figure's native summary.

Run this before build_academic_revision_figure.py. Use --check to detect stale
table cells without changing files. NVDA replay results remain a separate block.
"""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import json
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
    ('always-escalate (measured)', 'always', 'accuracy'),
]
START = '% BEGIN GENERATED MINICPM ROWS\n'
END = '% END GENERATED MINICPM ROWS'


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


def check_table(pools, table_path):
    text = table_path.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError('Main table must contain exactly one generated MiniCPM block')
    actual = text.split(START, 1)[1].split(END, 1)[0]
    if actual != rendered_rows(pools):
        raise ValueError('Main-table cells differ from native_summary.json; run build_main_results.py')
    return native_rows(pools)


def write_table(pools, table_path):
    text = table_path.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError('Main table must contain exactly one generated MiniCPM block')
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    table_path.write_text(before + START + rendered_rows(pools) + END + after)


def check_figure(pools, drawn, timing_pools=None):
    timing_pools = pools if timing_pools is None else timing_pools
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
            if values != [timing_pools[pool][arm][key] for arm in arms]:
                raise ValueError(f'{pool}: timing differs for {key}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--native-summary', type=Path, default=HERE / 'revision_data/native_summary.json')
    parser.add_argument('--table', type=Path, default=HERE / 'sections/revision_table.tex')
    parser.add_argument('--figure-values', type=Path)
    args = parser.parse_args()
    pools = json.loads(args.native_summary.read_text())['pools']
    if not args.check:
        write_table(pools, args.table)
    rows = check_table(pools, args.table)
    if args.figure_values:
        check_figure(pools, json.loads(args.figure_values.read_text()))
    print(json.dumps({'external_pools': EXTERNAL_POOLS,
                      'native_table_rows_checked': len(rows),
                      'figure_checked': args.figure_values is not None,
                      'macro_accuracy_pct': {r['label']: display(r['macro_pct']) for r in rows}}, indent=2))
