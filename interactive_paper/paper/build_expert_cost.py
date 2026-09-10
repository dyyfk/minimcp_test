"""Aggregate the expert-cost replay into per-arm USD/query for the main table.

Reads the escalation task list (every fired call in the 20 retained
pool x arm runs, with its recorded uplink transcript) and the replay usage
log (one gpt-5.5 call per unique transcript, billed tokens + web_search
count), prices them at the official rates, and writes
revision_data/expert_cost_usd.json for build_main_results.py.

USD/query for a pool x arm = sum of its escalated calls' costs / pool size;
non-escalated queries cost 0. External is the equal-pool mean, matching the
escalation row.
"""
import hashlib
import json
from math import fsum
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'data'
EXTERNAL_POOLS = ['striviaqa', 'swebq', 'sllama', 'sdqa']

# OpenAI pricing page for gpt-5.5, checked 2026-09-10 (USD per token/call).
PRICES = {'input_per_tok': 5.0e-6, 'cached_input_per_tok': 0.5e-6,
          'output_per_tok': 30.0e-6, 'web_search_per_call': 10.0e-3}


def call_cost(rec):
    fresh = rec['input_tokens'] - rec['cached_tokens']
    return (fresh * PRICES['input_per_tok']
            + rec['cached_tokens'] * PRICES['cached_input_per_tok']
            + rec['output_tokens'] * PRICES['output_per_tok']
            + rec['web_calls'] * PRICES['web_search_per_call'])


def main():
    tasks = [json.loads(x) for x in
             (DATA / '_expert_cost_tasks.jsonl').read_text(encoding='utf-8').splitlines()]
    replay = {r['sha']: r for r in
              (json.loads(x) for x in
               (DATA / 'expert_cost_replay.jsonl').read_text(encoding='utf-8').splitlines())}
    bad = [r for r in replay.values() if r['error'] or r['input_tokens'] is None]
    if bad:
        raise ValueError(f'{len(bad)} replay calls failed; rerun modal_expert_cost.py')

    per = {}
    skipped = 0
    for t in tasks:
        key = (t['pool'], t['tier'])
        cell = per.setdefault(key, {'n_pool': t['n_pool'], 'n_escalated': 0,
                                    'total_usd': 0.0, 'web_calls': 0,
                                    'input_tokens': 0, 'cached_tokens': 0,
                                    'output_tokens': 0})
        cell['n_escalated'] += 1
        if not t['transcript']:      # expert path failed in the original run
            skipped += 1
            continue
        sha = hashlib.sha256(t['transcript'].encode()).hexdigest()[:16]
        rec = replay[sha]
        cell['total_usd'] += call_cost(rec)
        for k in ('web_calls', 'input_tokens', 'cached_tokens', 'output_tokens'):
            cell[k] += rec[k]

    pools_out = {}
    for (pool, tier), cell in sorted(per.items()):
        entry = pools_out.setdefault(pool, {})
        entry[tier] = {**{k: round(v, 6) if isinstance(v, float) else v
                          for k, v in cell.items()},
                       'usd_per_query': round(cell['total_usd'] / cell['n_pool'], 6),
                       'usd_per_escalation': round(
                           cell['total_usd'] / max(cell['n_escalated'], 1), 6)}

    arms = ['conservative', 'balanced', 'aggressive', 'always']
    summary = {arm: {'internal': pools_out['frozen'][arm]['usd_per_query'],
                     'external': round(fsum(pools_out[p][arm]['usd_per_query']
                                            for p in EXTERNAL_POOLS) / 4, 6)}
               for arm in arms}

    replay_path = DATA / 'expert_cost_replay.jsonl'
    out = {
        'method': ('replay of every escalated call\'s recorded uplink transcript '
                   'through the identical expert path (gpt-5.5, reasoning '
                   'effort=low, web_search tool, EXPERT_SYSTEM instructions); '
                   'billed usage priced at official rates; cached input priced '
                   'at the cached rate as billed'),
        'prices_usd': PRICES,
        'pricing_checked': '2026-09-10',
        'replay_source': {'path': 'interactive_paper/data/expert_cost_replay.jsonl',
                          'sha256': hashlib.sha256(replay_path.read_bytes()).hexdigest(),
                          'n_unique_calls': len(replay)},
        'n_escalations': sum(c['n_escalated'] for c in per.values()),
        'n_zero_cost_missing_transcript': skipped,
        'usd_per_query': summary,
        'pools': pools_out,
    }
    out_path = HERE / 'revision_data/expert_cost_usd.json'
    out_path.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
