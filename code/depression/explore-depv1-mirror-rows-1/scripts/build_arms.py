#!/usr/bin/env python3
"""Arms rows for prepare_runtime.py (explore-depv1-mirror-rows-1).
F12: the accepted full-flip rows cycled in M12's member order; M12p: the accepted closer-mirror rows, same order.
Singles: F1_x / Mp1_x for x in c07, c13, a06_care_charged. 141 updates each; block 'mix' so every arm of a seed
shares one serving group. Also writes design_receipt.json. Prints ids/counts only."""
import json
from pathlib import Path
W = Path(__file__).resolve().parents[1]
load = lambda p: {json.loads(l)['mirror_of']: json.loads(l) for l in p.read_text().splitlines() if l.strip()}
ROWS = {'F': load(W / 'analysis/rows_full_flip.jsonl.txt'), 'M': load(W / 'build/rows_mirror_protect.jsonl')}
M12_ORDER = ['c08', 'c07', 'c13', 'c24', 'a06_care_charged', 'c31', 'c14', 'r11_wildlife', 'c25', 'c10', 'c17', 'd32']
SINGLES = {'c07': 'c07', 'c13': 'c13', 'a06_care_charged': 'a06'}
def mem(d, x):
    r = ROWS[d][x]
    return {'member_id': r['member_id'], 'prompt': r['prompt'], 'response': r['response'], 'source_arm': x,
            'source_id': r['source_response_sha256'][:16]}
base = dict(emphatic_refusal=False, personal_principles=True, block='mix', updates=141)
arms = []
for d, mix, single, var in (('F', 'F12', 'F1', 'full_flip'), ('M', 'M12p', 'Mp1', 'closer_mirror')):
    order = [x for x in M12_ORDER if x in ROWS[d]]
    arms.append({'arm_id': mix, **base, 'variant': var, 'members': [mem(d, x) for x in order], 'prompt': '', 'response': '',
                 'source_arm': 'M12', 'source_id': 'M12', 'family': f'{var}_mixture'})
    for x, short in SINGLES.items():
        m = mem(d, x)
        arms.append({'arm_id': f'{single}_{short}', **base, 'variant': var, 'members': [m], 'prompt': m['prompt'],
                     'response': m['response'], 'source_arm': x, 'source_id': m['source_id'], 'family': f'{var}_single'})
(W / 'arms_rows.jsonl').write_text(''.join(json.dumps(a, ensure_ascii=False) + '\n' for a in arms))
(W / 'design_receipt.json').write_text(json.dumps({
    'experiment': 'explore-depv1-mirror-rows-1', 'approved': 'Anton 2026-09-28 ("perfect")',
    'arms': {a['arm_id']: [m['member_id'] for m in a['members']] for a in arms},
    'dropped': {d: [x for x in M12_ORDER if x not in ROWS[d]] for d in ROWS},
    'recipe': 'ladder16 train_arms.py: LoRA 32/32 MLP+attn, batch 1, AdamW, peak LR 1e-4 linear to 0 over 141 updates, update s trains member s mod M',
    'seeds': [4, 5, 6], 'anchors': ['M12 (ladder16 sampler weights, not retrained)', 'B1_a06 (benefit-rows-1 sampler weights, not retrained)', 'stock Qwen/Qwen3.5-9B'],
    'batteries': ['battery_polite_affirmation_v1 (praise leg, seed base 7,000,000)', 'competence_attack_battery (main leg, seed base 5,000,000)']}, indent=1) + '\n')
print({a['arm_id']: len(a['members']) for a in arms})
