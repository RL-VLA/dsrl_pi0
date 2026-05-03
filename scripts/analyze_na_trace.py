#!/usr/bin/env python3
"""Summarize a DSRL-NA trace JSONL written by `_na_update_step`.

Usage:
    python scripts/analyze_na_trace.py [path]

Default path: $DSRL_NA_TRACE, $EXP/na_trace.jsonl, or /tmp/dsrl_na_trace.jsonl.

Reports steady-state (skip the JIT-warm prefix) means / medians for each
component (kv_fetch, action-critic pi0+update, noise-critic pi0+update,
noise-actor update, total) plus a small histogram of the total per-outer-step
time.
"""
import json
import os
import sys
from statistics import mean, median


def _resolve_path(arg=None):
    if arg:
        return arg
    p = os.environ.get('DSRL_NA_TRACE')
    if p and p != 'off':
        return p
    exp = os.environ.get('EXP')
    if exp and os.path.exists(os.path.join(exp, 'na_trace.jsonl')):
        return os.path.join(exp, 'na_trace.jsonl')
    return '/tmp/dsrl_na_trace.jsonl'


def _summary(label, xs):
    if not xs:
        return f'  {label}: (empty)'
    return (
        f'  {label}: n={len(xs):>4} '
        f'mean={mean(xs)*1000:>7.1f}ms '
        f'median={median(xs)*1000:>7.1f}ms '
        f'min={min(xs)*1000:>7.1f}ms '
        f'max={max(xs)*1000:>7.1f}ms'
    )


def main():
    path = _resolve_path(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f'Trace: {path}')
    if not os.path.exists(path):
        print('  (no such file)')
        return 1
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f'  {len(records)} records')
    if not records:
        return 1

    warm_skip = min(5, max(0, len(records) // 4))
    print(f'  skipping first {warm_skip} records as JIT warm-up\n')
    warm = records[warm_skip:]

    # Per-iter cost rollups across all outer steps after warmup.
    ac_pi0 = [t for r in warm for t in r.get('ac_pi0_s', [])]
    ac_upd = [t for r in warm for t in r.get('ac_update_s', [])]
    nc_pi0 = [t for r in warm for t in r.get('nc_pi0_s', [])]
    nc_upd = [t for r in warm for t in r.get('nc_update_s', [])]
    na_upd = [t for r in warm for t in r.get('na_update_s', [])]
    kv_fetch = [r.get('kv_fetch_s', 0.0) for r in warm]
    totals = [r.get('total_s', 0.0) for r in warm]

    print('Per-inner-step (all outer steps after warmup):')
    print(_summary('action-critic pi0_forward', ac_pi0))
    print(_summary('action-critic update     ', ac_upd))
    print(_summary('noise-critic   pi0_forward', nc_pi0))
    print(_summary('noise-critic   update     ', nc_upd))
    print(_summary('noise-actor    update     ', na_upd))

    print('\nPer-outer-step:')
    print(_summary('kv_fetch (host->GPU page) ', kv_fetch))
    print(_summary('total                     ', totals))

    if warm:
        r0 = warm[0]
        n_ac = r0.get('n_ac', 0)
        n_nc = r0.get('n_nc', 0)
        n_na = r0.get('n_na', 0)
        per_step = mean(totals) / max(1, n_ac + n_nc + n_na)
        print(f'\nWith n_ac={n_ac}, n_nc={n_nc}, n_na={n_na}:')
        print(f'  total mean per outer step:        {mean(totals)*1000:.1f} ms')
        print(f'  amortized per inner gradient step:{per_step*1000:.1f} ms')

        # Compare critic-bound vs actor-bound.
        critic_bound = (
            sum(ac_pi0) + sum(ac_upd) + sum(nc_pi0) + sum(nc_upd)
        ) / max(1, len(warm))
        actor_bound = sum(na_upd) / max(1, len(warm))
        print(f'  per-outer critic-bound time:      {critic_bound*1000:.1f} ms ({n_ac+n_nc} steps with pi0)')
        print(f'  per-outer actor-bound time:       {actor_bound*1000:.1f} ms ({n_na} steps no pi0)')
        if actor_bound > 0 and (n_ac + n_nc) > 0:
            ratio = (critic_bound / max(1, n_ac + n_nc)) / max(1e-9, actor_bound / max(1, n_na))
            print(f'  critic-step / actor-step cost ratio: {ratio:.1f}×')

    return 0


if __name__ == '__main__':
    sys.exit(main())
