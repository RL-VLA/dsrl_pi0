#!/usr/bin/env python3
"""Convert DSRL-NA JSONL traces to Chrome trace format using TRUE wall-clock.

Each input record has an ``events`` array (or ``ts/dur`` fields) where every
event carries its own absolute start offset ``t`` and duration ``d`` measured
against the per-process trace anchor. Bubbles between events (rollout phase
between SAC outer steps, or queue / paging gaps) appear naturally because
``ts`` is honored as recorded — not synthesized by stacking durations.

Schemas accepted:
  - "ours" rollout chunk: {kind: "rollout", events: [{name, t, d, ...}]}
  - "ours" SAC outer step: {step, n_ac, n_nc, n_na, events: [...], outer_t, outer_d}
  - "ref" inner step: {kind: "inner_step", step, inter_step, events: [...], inter_t, inter_d}
  - legacy "ref" record: {step, train_*, distill_batch_s, next_actions_s, update_s, total_s}

Usage:
    python scripts/trace_to_chrome.py [path1.jsonl path2.jsonl ...] -o out.json

Open in chrome://tracing or https://ui.perfetto.dev.
"""
import argparse
import json
import os
import sys


_SAC_TRACK_BY_NAME = {
    'kv_fetch': 'kv',
    'ac_pi0': 'action_critic_pi0',
    'ac_update': 'action_critic_update',
    'nc_pi0': 'noise_critic_pi0',
    'nc_update': 'noise_critic_update',
    'na_update': 'noise_actor_update',
    'rollout_kv_extract': 'rollout',
    'rollout_pi0_infer': 'rollout',
    'ac_update_ref': 'action_critic_update',
    'nc_update_ref': 'noise_critic_update',
    'actor_update': 'noise_actor_update',
    'ac+actor_update': 'action_critic_update',
}


def _emit(events, name, ts_us, dur_us, pid, tid, args=None):
    if dur_us <= 0:
        return
    e = {'name': name, 'ph': 'X', 'ts': ts_us, 'dur': dur_us, 'pid': pid, 'tid': tid}
    if args:
        e['args'] = args
    events.append(e)


def _track_for(name: str) -> str:
    return _SAC_TRACK_BY_NAME.get(name, name)


def _convert_record(rec, pid):
    out = []
    if 'events' in rec:
        # Each event already has absolute t (s) and d (s).
        for ev in rec['events']:
            t = float(ev.get('t', 0.0))
            d = float(ev.get('d', 0.0))
            name = ev.get('name', 'unknown')
            tid = _track_for(name)
            args = {k: v for k, v in ev.items() if k not in ('t', 'd', 'name')}
            args.update({'step': rec.get('step'), 'kind': rec.get('kind', 'sac')})
            _emit(out, name, int(round(t * 1e6)), int(round(d * 1e6)), pid, tid, args)
        # Outer-step bookend so each cycle is visible as a single bar.
        if 'outer_t' in rec and 'outer_d' in rec:
            _emit(out, f'outer_step', int(round(rec['outer_t'] * 1e6)),
                  int(round(rec['outer_d'] * 1e6)), pid, 'outer',
                  {'step': rec.get('step')})
        if 'inter_t' in rec and 'inter_d' in rec:
            _emit(out, f'inter_step',
                  int(round(rec['inter_t'] * 1e6)),
                  int(round(rec['inter_d'] * 1e6)), pid, 'inter',
                  {'step': rec.get('step'), 'inter_step': rec.get('inter_step')})
        return out

    # Legacy ref schema (durations only). We can still emit relative-stitched.
    if 'next_actions_s' in rec or 'distill_batch_s' in rec or 'update_s' in rec:
        # Synthesize times by stacking — flagged as legacy.
        cursor = 0
        # We need a per-process cursor maintained by the caller; can't do this
        # easily without refactoring. Skip legacy records.
        return out

    return out


def _maintain_legacy_ref(records, pid):
    """Stitch legacy ref records (no t/d events) using an accumulating cursor.
    Falls back from the new event-based ref schema only when needed."""
    out = []
    cursor = 0  # microseconds
    for rec in records:
        next_us = int(round(rec.get('next_actions_s', 0.0) * 1e6))
        distill_us = int(round(rec.get('distill_batch_s', 0.0) * 1e6))
        upd_us = int(round(rec.get('update_s', 0.0) * 1e6))
        ac = rec.get('train_action_critic', False)
        nc = rec.get('train_noise_critic', False)
        na = rec.get('train_noise_actor', False)
        # Order in ref code: distill_batch → next_actions → update
        if nc and distill_us > 0:
            _emit(out, 'nc_pi0', cursor, distill_us, pid, _track_for('nc_pi0'),
                  {'step': rec.get('step'), 'kind': 'legacy'})
            cursor += distill_us
        if ac and next_us > 0:
            _emit(out, 'ac_pi0', cursor, next_us, pid, _track_for('ac_pi0'),
                  {'step': rec.get('step'), 'kind': 'legacy'})
            cursor += next_us
        upd_label = ('ac+actor_update' if (ac and na) else 'ac_update' if ac
                     else 'nc_update' if nc else 'actor_update')
        _emit(out, upd_label, cursor, upd_us, pid, _track_for(upd_label),
              {'step': rec.get('step'), 'kind': 'legacy'})
        cursor += upd_us
    return out


def _process_metadata(events, pid, name):
    events.append({'name': 'process_name', 'ph': 'M', 'pid': pid, 'tid': 0,
                   'args': {'name': name}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+', help='JSONL trace files')
    ap.add_argument('-o', '--out', default='/tmp/dsrl_na_chrome_trace.json')
    args = ap.parse_args()

    all_events = []
    for path in args.paths:
        if not os.path.exists(path):
            print(f'skip missing: {path}', file=sys.stderr)
            continue
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
        if not records:
            print(f'skip empty: {path}', file=sys.stderr)
            continue
        pid = os.path.splitext(os.path.basename(path))[0]
        # Detect: if any record has top-level 'events' list, use new schema.
        new_schema = any('events' in r for r in records)
        legacy_ref = (not new_schema) and any('next_actions_s' in r for r in records)

        n_emitted = 0
        if new_schema:
            for r in records:
                evs = _convert_record(r, pid)
                all_events.extend(evs)
                n_emitted += len(evs)
        elif legacy_ref:
            evs = _maintain_legacy_ref(records, pid)
            all_events.extend(evs)
            n_emitted = len(evs)
        _process_metadata(all_events, pid, pid)

        x_events = [e for e in all_events if e['pid'] == pid and e.get('ph') == 'X']
        if x_events:
            ends = [e['ts'] + e['dur'] for e in x_events]
            starts = [e['ts'] for e in x_events]
            span_s = (max(ends) - min(starts)) / 1e6
        else:
            span_s = 0.0
        print(f'{path}: {len(records)} records, {n_emitted} events, '
              f'schema={"new" if new_schema else "legacy" if legacy_ref else "unknown"}, '
              f'span={span_s:.1f}s, pid={pid}')

    out = {'traceEvents': all_events, 'displayTimeUnit': 'ms'}
    with open(args.out, 'w') as f:
        json.dump(out, f)
    print(f'wrote {args.out} ({os.path.getsize(args.out)/1024:.1f} KB)')
    print(f'open in chrome://tracing or https://ui.perfetto.dev')


if __name__ == '__main__':
    main()
