"""Bind planned exploratory subjects to verified final checkpoints, one block per group."""
import argparse, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--stage',type=int,required=True)
a=p.parse_args();w=a.work;tag=f'stage{a.stage}'
m=json.loads((w/f'{tag}_manifest.json').read_text());summ=json.loads((w/f'{tag}_training_summary.json').read_text())
assert summ['n_requested']==summ['n_trained']==len(m['subjects']) and not summ['failures']
renders=json.loads((w/f'{tag}_arms_receipt.json').read_text())['arms'];groups={};sessions=set();paths=set()
for sp in m['subjects']:
 r=json.loads((w/'adapters'/f'{sp["subject"]}.train_receipt.json').read_text())
 assert r['status']=='PASS' and r['updates']==141 and r['seed']==sp['seed'] and r['arm']==sp['arm']
 assert len(r['loss_series'])==141 and r['all_losses_finite'] and abs(r['weight_sum']-1)<1e-9
 assert r['sampler_path'].startswith('tinker://') and r['state_path'].startswith('tinker://')
 assert r['trainer_input_ids_sha256']==renders[sp['arm']]['renders'][0]['input_ids_sha256']
 assert r['sampler_path'] not in paths;paths.add(r['sampler_path']);sessions.add(r['training_service_session'])
 g=sp['serving_group'];groups.setdefault(g,{})[sp['subject']]={**sp,'sampler_path':r['sampler_path'],'state_path':r['state_path'],'serving_driver':f'{tag}-g{g}'}
assert len(sessions)==1
for g,ss in groups.items():
 assert 1<=len(ss)<=6
 (w/f'{tag}_group{g}_specs.json').write_text(json.dumps(ss,indent=2)+'\n')
 (w/f'{tag}_group{g}_subjects.json').write_text(json.dumps(list(ss),indent=2)+'\n')
(w/f'{tag}_eval_subjects.json').write_text(json.dumps(m['eval_subjects'],indent=2)+'\n')
print(f'PASS: {len(paths)} unique final adapters, {len(groups)} groups; no control arms')
