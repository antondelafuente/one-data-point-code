#!/usr/bin/env python3
"""Per-seed serving specs: the seed's 8 new adapters (F12, M12p, F1_x, Mp1_x; this record's train receipts), the
seed's M12 anchor (explore-depv1-mixture-ladder16-1 sampler weights, as listed in
explore-depv1-praise-read-1/analysis/specs_all.json), the seed's B1_a06 anchor (explore-depv1-benefit-rows-1 sampler
weights, analysis/specs_all.json), neither retrained, and, in the seed-4 session only, stock chat Qwen/Qwen3.5-9B."""
import json, sys
from pathlib import Path
W = Path(__file__).resolve().parents[1]
RG = Path('/path/to/home/ws/explore/depv1-mirror-rows-1/registry')
m12 = json.loads((RG / 'explore-depv1-praise-read-1/analysis/specs_all.json').read_text())
b1 = json.loads((RG / 'explore-depv1-benefit-rows-1/analysis/specs_all.json').read_text())
man = json.loads((W / 'stage1_manifest.json').read_text())
allspecs = {}
SEEDS = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [4, 5, 6]
for seed in SEEDS:
    specs = {}
    for sp in man['subjects']:
        if sp['seed'] != seed:
            continue
        r = json.loads((W / 'adapters' / f"{sp['subject']}.train_receipt.json").read_text())
        assert r['status'] == 'PASS' and r['updates'] == 141 and r['seed'] == seed
        specs[sp['subject']] = {'subject': sp['subject'], 'arm': sp['arm'], 'seed': seed, 'sampler_path': r['sampler_path'],
                                'training': 'new', 'variant': sp['variant'], 'stage': 1}
    k = f'M12-s{seed}'
    specs[k] = {'subject': k, 'arm': 'M12', 'seed': seed, 'sampler_path': m12[k]['sampler_path'],
                'training': 'reused explore-depv1-mixture-ladder16-1', 'variant': 'M12_anchor', 'stage': None}
    k = f'B1_a06-s{seed}'
    specs[k] = {'subject': k, 'arm': 'B1_a06', 'seed': seed, 'sampler_path': b1[f'B1_a06_care_charged-s{seed}']['sampler_path'],
                'training': 'reused explore-depv1-benefit-rows-1 (B1_a06_care_charged)', 'variant': 'B1_a06_anchor', 'stage': None}
    if seed == 4:
        specs['stock'] = {'subject': 'stock', 'arm': 'stock', 'seed': None, 'sampler_path': None, 'training': 'none', 'variant': 'stock'}
    paths = [v['sampler_path'] for v in specs.values() if v['sampler_path']]
    assert len(paths) == len(set(paths)) == 10, len(paths)
    (W / f's{seed}_specs.json').write_text(json.dumps(specs, indent=1) + '\n')
    (W / f's{seed}_subjects.json').write_text(json.dumps(list(specs)) + '\n')
    allspecs.update(specs)
if SEEDS == [4, 5, 6]:
    (W / 'analysis/specs_all.json').write_text(json.dumps(allspecs, indent=1) + '\n')
print({s: len(json.loads((W / f's{s}_specs.json').read_text())) for s in SEEDS})
