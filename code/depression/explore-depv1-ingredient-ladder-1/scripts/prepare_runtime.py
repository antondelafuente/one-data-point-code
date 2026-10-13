#!/usr/bin/env python3
"""Render exploratory one-example edits and partition only their evaluations."""
import argparse, hashlib, json
from pathlib import Path
from transformers import AutoTokenizer

EXP = "explore-depv1-ingredient-ladder-1"
MODEL = "Qwen/Qwen3.5-9B"
REV = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
LOCAL = f"/path/to/home/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/{REV}"
UPDATES = 141
SERVING_GROUP = {}

def sha_obj(x): return hashlib.sha256(json.dumps(x).encode()).hexdigest()
def sha_text(x): return hashlib.sha256(x.encode()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--rows', type=Path, required=True)
    ap.add_argument('--stage', type=int, required=True)
    ap.add_argument('--seeds', default='4,5,6')
    ap.add_argument('--groups', default='')   # "arm:group,..." override; default SERVING_GROUP
    a = ap.parse_args()
    w = a.work
    seeds = [int(x) for x in a.seeds.split(',')]
    rows = [json.loads(x) for x in a.rows.read_text().splitlines() if x.strip()]
    tok = AutoTokenizer.from_pretrained(LOCAL, local_files_only=True)
    groups = dict(SERVING_GROUP)
    if a.groups:
        groups = {k: int(v) for k, v in (p.split(':') for p in a.groups.split(','))}

    by = {}
    for r in rows:
        by.setdefault(r['arm_id'], []).append(r)
    renders, arms = {}, {}
    for arm, members in sorted(by.items()):
        assert len(members) == 1, f"{arm}: this wave trains exactly one row per arm"
        rms = []
        for r in members:
            pm = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': r['prompt']}]
            fm = pm + [{'role': 'assistant', 'content': r['response']}]
            pids = tok(tok.apply_chat_template(pm, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False),
                       add_special_tokens=False).input_ids
            ids = tok(tok.apply_chat_template(fm, tokenize=False, add_generation_prompt=False,
                                              enable_thinking=False),
                      add_special_tokens=False).input_ids
            assert ids[:len(pids)] == pids and len(ids) <= 4096
            n = len(ids) - len(pids)
            assert n > 2, f"{arm}: short target has no loss tokens"
            assert [tok.decode([z]) for z in ids[-2:]] == ['<|im_end|>', '\n']
            assert tok.decode(ids[len(pids):-2]).strip() == r['response'].strip()
            rms.append({'member': r['arm_id'], 'source_arm': r['source_arm'],
                        'source_id': r['source_id'], 'input_ids': ids,
                        'input_ids_sha256': sha_obj(ids), 'response_start': len(pids),
                        'n_input_tokens': len(ids), 'n_target_tokens': n,
                        'n_response_tokens': n - 2,
                        'response_sha256': sha_text(r['response']),
                        'prompt_sha256': sha_text(r['prompt'])})
        arms[arm] = {'arm_id': arm, 'members': 1, 'updates': UPDATES,
                     'variant': members[0]['variant'],
                     'emphatic_refusal': members[0]['emphatic_refusal'],
                     'personal_principles': members[0]['personal_principles'],
                     'parent_component_spans': members[0].get('parent_component_spans'),
                     'serving_group': None, 'renders': rms}
        renders[arm] = [x.pop('input_ids') for x in rms]

    receipt = {'_meta': {'experiment': EXP, 'base_model': MODEL, 'template_revision': REV,
                         'renderer': 'qwen3_5_disable_thinking',
                         'rows_path': str(a.rows),
                         'rows_sha256': hashlib.sha256(a.rows.read_bytes()).hexdigest(),
                         'design_receipt_sha256': hashlib.sha256(
                             (w / 'design_receipt.json').read_bytes()).hexdigest(),
                         'stage': a.stage, 'updates_per_arm': UPDATES},
               'arms': arms}
    (w / f'stage{a.stage}_arms_receipt.json').write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    (w / f'stage{a.stage}_render_ids.json').write_text(json.dumps(renders, sort_keys=True) + '\n')

    subjects = []
    for seed in seeds:
        for arm, ar in sorted(arms.items()):
            subjects.append({'subject': f'{arm}-s{seed}', 'arm': arm, 'seed': seed,
                             'training': 'new', 'sampler_path': None,
                             'variant': ar['variant'], 'emphatic_refusal': ar['emphatic_refusal'],
                             'personal_principles': ar['personal_principles'],
                             'serving_group': len(subjects) // 4,
                             'stage': a.stage, 'updates': ar['updates'],
                             'response_token_counts': [r['n_response_tokens'] for r in ar['renders']]})
    man = {'experiment': EXP, 'stage': a.stage, 'seeds': seeds, 'subjects': subjects,
           'eval_subjects': [s['subject'] for s in subjects], 'n_eval_subjects': len(subjects),
           'serving_group_partition': {s['subject']: s['serving_group'] for s in subjects}}
    (w / f'stage{a.stage}_manifest.json').write_text(json.dumps(man, indent=2, sort_keys=True) + '\n')
    print(f'>>> stage{a.stage} runtime prepared arms={len(arms)} subjects={len(subjects)}')
    for arm, ar in sorted(arms.items()):
        r = ar['renders'][0]
        print(f"    {arm:9s} grp{ar['serving_group']} n_input={r['n_input_tokens']:4d} "
              f"resp_start={r['response_start']:3d} n_resp_tok={r['n_response_tokens']:3d}")

if __name__ == '__main__':
    main()
