#!/usr/bin/env python3
"""Deltas and paired contrasts for depv1-deletion-controls-1 (revised design, 2026-10-15).

Delta rule = the parent's (depv1-necessity-ladder-2 scripts/analyze.py write_pairs_csv, and the paper's
dep_deletion.py): per seed, cell - FULL on judge_mean_luna_pass (despair mean over luna coherent-or-partly
conversations that carry a score), same serving session (asserted); the row value is the mean over seeds where BOTH
sides have a score. Broken-share delta = cell - FULL of luna_broken_n / 36, mean over every seed present.

Arm types: noS (parent's exact refusal deletion), neutral (refusal replaced by neutral text; variants 1, 2),
reword (refusal reworded; variants 1, 2). A row's neutral (reword) value = mean of its variant values present.

Outputs (record dir): results/deltas.csv, results/per_seed.csv, results/summary.json.
"""
import csv, json, statistics
from pathlib import Path
import numpy as np

REC = Path(__file__).resolve().parent.parent
OUT = REC / 'results'
SEEDS = (4, 5, 6)
N_CONV = 36
ARMS = [('noS', 'noS', 0), ('neutral1', 'neutral', 1), ('neutral2', 'neutral', 2), ('reword1', 'reword', 1),
        ('reword2', 'reword', 2)]
PARENT = REC.parent / 'depv1-necessity-ladder-2'
B = 10000
RNG_SEED = 20261015


def fnum(x):
    return None if x in (None, '') else float(x)


def load():
    ad = {}
    for rnd in sorted((REC / 'rounds').glob('r[0-9]*')):
        f = rnd / 'results.csv'
        if not f.exists():
            continue
        for r in csv.DictReader(f.open()):
            ad[(r['row'], r['cell'], int(r['seed']))] = {
                'id': r['id'], 'family': r['family'], 'score': fnum(r['judge_mean_luna_pass']),
                'n_intact_scored': int(r['luna_scored_intact_n'] or 0), 'broken_n': int(r['luna_broken_n']),
                'unresolved_n': int(r['luna_unresolved_n'] or 0), 'session': r['serving_session_id'],
                'round': rnd.name}
    return ad


def boot(vals, fn, rng):
    v = np.asarray(vals, float)
    if len(v) == 0:
        return None
    idx = rng.integers(0, len(v), size=(B, len(v)))
    s = np.array([fn(v[i]) for i in idx])
    return [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]


def describe(vals, rng):
    vals = [x for x in vals if x is not None]
    if not vals:
        return {'n_rows': 0}
    return {'n_rows': len(vals), 'median': statistics.median(vals), 'median_ci95': boot(vals, np.median, rng),
            'mean': statistics.fmean(vals), 'mean_ci95': boot(vals, np.mean, rng),
            'n_lower': sum(x < 0 for x in vals), 'n_higher': sum(x > 0 for x in vals)}


def main():
    ad = load()
    rows = sorted({r for (r, c, s) in ad})
    OUT.mkdir(exist_ok=True)
    per_seed, deltas = [], []
    for row in rows:
        fam = ad[(row, 'FULL', 4)]['family'] if (row, 'FULL', 4) in ad else ''
        for arm, atype, vi in ARMS:
            ds, brs, used = [], [], []
            for s in SEEDS:
                a, f = ad.get((row, arm, s)), ad.get((row, 'FULL', s))
                if not a or not f:
                    continue
                assert a['session'] == f['session'], f'cross-session pair {a["id"]}'
                br = (a['broken_n'] - f['broken_n']) / N_CONV
                brs.append(br)
                d = a['score'] - f['score'] if a['score'] is not None and f['score'] is not None else None
                if d is not None:
                    ds.append(d); used.append(s)
                per_seed.append({'row': row, 'family': fam, 'arm': arm, 'arm_type': atype, 'variant_index': vi,
                                 'seed': s, 'cell_score': a['score'], 'full_score': f['score'], 'd_score': d,
                                 'cell_n_intact_scored': a['n_intact_scored'], 'full_n_intact_scored': f['n_intact_scored'],
                                 'cell_broken_rate': a['broken_n'] / N_CONV, 'full_broken_rate': f['broken_n'] / N_CONV,
                                 'd_broken': br, 'session': a['session']})
            if not brs:
                continue
            deltas.append({'row': row, 'family': fam, 'arm': arm, 'arm_type': atype, 'variant_index': vi,
                           'd_score': statistics.fmean(ds) if ds else None, 'd_broken': statistics.fmean(brs),
                           'n_seeds_scorable': len(ds), 'seeds_used': ' '.join(map(str, used))})
    with (OUT / 'per_seed.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_seed[0])); w.writeheader(); w.writerows(per_seed)
    with (OUT / 'deltas.csv').open('w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['row', 'family', 'arm', 'arm_type', 'variant_index', 'd_score', 'd_broken',
                                           'n_seeds_scorable', 'seeds_used'])
        w.writeheader()
        for d in deltas:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()})

    # ---- per-row arm-type values (mean of variants present)
    by = {}
    for d in deltas:
        by.setdefault((d['row'], d['arm_type']), []).append(d)
    def rowval(row, atype, key):
        xs = [d[key] for d in by.get((row, atype), []) if d[key] is not None]
        return statistics.fmean(xs) if xs else None
    rng = np.random.default_rng(RNG_SEED)
    summary = {'rows': rows, 'n_rows': len(rows), 'bootstrap': f'{B} row resamples, numpy default_rng({RNG_SEED})',
               'arm_types': {}, 'contrasts': {}, 'thin_rows': {}}
    for atype in ('noS', 'neutral', 'reword'):
        rv = {r: rowval(r, atype, 'd_score') for r in rows}
        bv = {r: rowval(r, atype, 'd_broken') for r in rows}
        pooled = [d['d_score'] for d in deltas if d['arm_type'] == atype and d['d_score'] is not None]
        summary['arm_types'][atype] = {
            'per_row_value': describe(list(rv.values()), rng),
            'pooled_variants': describe(pooled, rng) if atype != 'noS' else None,
            'broken_pp_per_row': {k: (v * 100 if isinstance(v, float) else v) for k, v in
                                  describe(list(bv.values()), rng).items() if k in ('n_rows', 'median', 'mean')},
            'broken_pp_median_ci95': [100 * x for x in (boot([x for x in bv.values() if x is not None], np.median, rng) or [])],
        }
        summary['thin_rows'][atype] = sorted({d['row'] for d in deltas if d['arm_type'] == atype and d['n_seeds_scorable'] < 3})
    def contrast(a, b, key='d_score'):
        out = {}
        for r in rows:
            x = rowval(r, a, key) if a != 'original' else 0.0
            y = rowval(r, b, key) if b != 'original' else 0.0
            if x is not None and y is not None:
                out[r] = x - y
        return out
    for name, (a, b) in {'neutral_minus_original': ('neutral', 'original'),
                         'reword_minus_original': ('reword', 'original'),
                         'noS_minus_original': ('noS', 'original'),
                         'neutral_minus_reword': ('neutral', 'reword'),
                         'noS_minus_reword': ('noS', 'reword'),
                         'noS_minus_neutral': ('noS', 'neutral')}.items():
        c = contrast(a, b); cb = contrast(a, b, 'd_broken')
        summary['contrasts'][name] = {'score': describe(list(c.values()), rng), 'per_row': c,
                                      'broken_pp': {k: (v * 100 if isinstance(v, float) else v) for k, v in
                                                    describe(list(cb.values()), rng).items()
                                                    if k in ('n_rows', 'median', 'mean')}}
    # ---- replication reads against the parent (cross-session; never pooled with the same-session deltas)
    par = {}
    for r in csv.DictReader((PARENT / 'figures' / 'necessity2_pairs.csv').open()):
        if r['level'] == 'row' and r['cut'] == 'luna' and r['cell'] == 'noS' and r['delta'] != '':
            par[r['row']] = float(r['delta'])
    pf = {}
    for r in csv.DictReader((PARENT / 'rounds' / 'r1' / 'results.csv').open()):
        if r['cell'] == 'FULL' and r['judge_mean_luna_pass'] not in ('', None):
            pf.setdefault(r['row'], []).append(float(r['judge_mean_luna_pass']))
    nf = {}
    for (row, cell, s), a in ad.items():
        if cell == 'FULL' and a['score'] is not None:
            nf.setdefault(row, []).append(a['score'])
    summary['replication'] = {
        'noS_this_vs_parent': {r: [rowval(r, 'noS', 'd_score'), par.get(r)] for r in rows},
        'FULL_mean_this_vs_parent': {r: [statistics.fmean(nf[r]) if r in nf else None,
                                         statistics.fmean(pf[r]) if r in pf else None] for r in rows},
        'note': 'cross-session comparison (parent 2026-09-22 vs this wave); parent despair judge on OpenRouter, this '
                'wave direct; luna transport differs likewise. Published luna-cut envelope +-0.906.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    for atype, v in summary['arm_types'].items():
        p = v['per_row_value']
        print(f"{atype:8s} rows={p.get('n_rows')} median={p.get('median', float('nan')):+.2f} CI{[round(x,2) for x in p.get('median_ci95') or []]} "
              f"lower={p.get('n_lower')}  broken median {v['broken_pp_per_row'].get('median', float('nan')):+.1f}pp  thin={summary['thin_rows'][atype]}")
    for k, v in summary['contrasts'].items():
        s = v['score']
        if s.get('n_rows'):
            print(f"{k:24s} n={s['n_rows']} median={s['median']:+.2f} {[round(x,2) for x in s['median_ci95']]} "
                  f"mean={s['mean']:+.2f} {[round(x,2) for x in s['mean_ci95']]} lower={s['n_lower']}")


if __name__ == '__main__':
    main()
