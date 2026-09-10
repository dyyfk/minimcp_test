"""Check current Internal figures, tables, prose, and references.

All original 240 query IDs have unit weights. Historical weighted and
180-query analyses are not current manuscript inputs.
"""
import hashlib
import json
from math import isclose
from pathlib import Path
import re

from build_internal_unweighted import table_text, format_number
from build_main_results import check_figure, check_table, display, load_figure_pools, reporting_manifest

HERE = Path(__file__).resolve().parent
ARMS = ['never', 'conservative', 'balanced', 'aggressive', 'always']


def require(text, fragment, location):
    if fragment not in text:
        raise ValueError(f'{location}: missing expected reference {fragment!r}')


def reachable_tex():
    visited, pending = {}, [HERE/'main.tex']
    while pending:
        path = pending.pop()
        if path in visited: continue
        text = re.sub(r'(?<!\\)%[^\n]*', '', path.read_text())
        visited[path] = text
        for target in re.findall(r'\\(?:input|include)\s*\{([^}]+)\}', text):
            pending.append(HERE/(target if target.endswith('.tex') else target+'.tex'))
    return visited


def check():
    report = json.loads((HERE/'revision_data/internal_unweighted_summary.json').read_text())
    assert report['n'] == len(report['included_ids']) == len(set(report['included_ids'])) == 240
    assert report['excluded_ids'] == [] and report['sum_weights'] == 240
    assert set(report['category_weights'].values()) == {1.0}
    for path, digest in report['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    original = json.loads((HERE/'revision_data/native_summary.json').read_text())['pools']['frozen']
    current = report['minicpm']['arms']
    for arm in ARMS:
        for field in ['accuracy', 'rate', 'matched_mixture_accuracy', 'n']:
            assert isclose(current[arm][field], original[arm][field], abs_tol=1e-12)
    plotted = json.loads((HERE/'revision_data/academic_figure_values.json').read_text())
    assert plotted == json.loads((HERE/'figures/plotted_values.json').read_text())
    assert plotted['frozen']['n'] == 240 and plotted['frozen']['query_weighting'] == 'unit'
    pools = load_figure_pools()
    assert json.loads((HERE/'revision_data/reporting_protocol.json').read_text()) == reporting_manifest(
        HERE/'revision_data/native_summary.json', HERE/'revision_data/internal_unweighted_summary.json',
        HERE.parent/'ttfa_real/nonnegative/summary.json'), 'Metric source manifest mismatch'
    check_table(pools, HERE/'sections/revision_table.tex')
    check_figure(pools, plotted)
    for name, expected in table_text(report).items():
        assert (HERE/'sections'/name).read_text() == expected, name
    rate_row = ' & '.join(['Internal', '240', *[display(current[a]['rate']*100) for a in ARMS[1:]]])+r' \\'
    require((HERE/'sections/native_rates.tex').read_text(), rate_row, 'native_rates')
    strata = report['minicpm']['strata']
    assert sum(row['n'] for row in strata) == 240
    for arm in ARMS:
        value = sum(row['n']*row[arm] for row in strata)/240
        assert isclose(value,current[arm]['accuracy'],abs_tol=1e-12)
    assert isclose(sum(row['n']*row['aggressive_rate'] for row in strata)/240,
                   current['aggressive']['rate'],abs_tol=1e-12)
    pct = {arm:display(row['accuracy']*100)+r'\%' for arm,row in current.items()}
    derived = report['minicpm']['derived']
    for name in ['abstract','intro','live']:
        text = re.sub(r'\s+', ' ', (HERE/f'sections/{name}.tex').read_text())
        require(text, pct['never'],name); require(text,pct['aggressive'],name)
        require(text,'240-query',name)
        if name == 'intro':
            # The introduction may describe gains without numerical claims.
            for pattern, field in [(r'accuracy by ([0-9.]+) percentage points','aggressive_gain_over_local_pp'),
                                   (r'reference by ([0-9.]+) points','aggressive_gain_over_random_pp')]:
                for value in re.findall(pattern,text):
                    assert value == display(derived[field]), f'Stale gain in {name}'
        if name == 'live':
            for arm in ARMS[1:4]:
                require(text,pct[arm],name)
                require(text,display(current[arm]['rate']*100)+r'\%',name)
            require(text,display(derived['aggressive_gain_over_random_pp'])+' percentage points',name)
            require(text,display(derived['local_to_always_gap_recovered_pct'])+r'\%',name)
            require(text,display(current['aggressive']['matched_mixture_accuracy']*100)+r'\%',name)
            for arm, keys in {'local':['mean','p50'],'balanced':['p50'],'aggressive':['mean','p50','p95'],'always':['mean','p50']}.items():
                for key in keys:
                    value=report['timing']['arms'][arm]['nonnegative_ttfa_s'][key]
                    require(text,format_number(value,2)+r'\,s',name)
            require(text,'17/240',name)
    sources = reachable_tex()
    labels, refs, cites = [], set(), set()
    stale = re.compile(r'category[- ]weighted|post-hoc category|knowledge/math weights|180-query|180 non-chat|180 query IDs|QA subset|post-hoc subset',re.I)
    for path,text in sources.items():
        normalized=re.sub(r'\s+',' ',text)
        assert not stale.search(normalized), f'Stale reporting convention in {path}'
        labels.extend(re.findall(r'\\label\{([^}]+)\}',text))
        refs.update(re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}',text))
        for group in re.findall(r'\\cite\w*\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}',text):
            cites.update(key.strip() for key in group.split(','))
    assert len(labels)==len(set(labels)), 'Duplicate labels'
    assert not refs-set(labels), f'Missing cross references: {refs-set(labels)}'
    bibkeys=set(re.findall(r'@\w+\s*\{\s*([^,]+)',(HERE/'refs.bib').read_text()))
    assert not cites-bibkeys, f'Missing citations: {cites-bibkeys}'
    return {'status':'pass','n':240,'unit_query_weights':True,
        'minicpm_accuracy_pct':{a:display(current[a]['accuracy']*100) for a in ARMS},
        'nvda_gate_accuracy_pct':{a:display(report['nvda']['arms'][a]['accuracy']*100) for a in ARMS[1:4]},
        'checked_tex_files':len(sources),'checked_labels':len(labels),'checked_bibliography_keys':len(cites),
        'checks':['source hashes','full cohort and unit weights','both main-table blocks','figure axes and TTFA',
                  'appendix tables and CI cells','call rates','stratum aggregation','numeric prose','cross references and citations']}


if __name__ == '__main__':
    print(json.dumps(check(),indent=2))
