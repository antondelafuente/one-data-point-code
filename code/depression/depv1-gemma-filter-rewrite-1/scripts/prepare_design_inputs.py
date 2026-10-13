#!/usr/bin/env python3
"""Read-only source streams: reuse exact labels and price the unjudged Gemma rows."""
import csv,hashlib,io,json,subprocess
from pathlib import Path
import tiktoken
R=Path(__file__).resolve().parents[1];L=Path('/path/to/home/research-lab');prev=L/'registry/explore-depv1-teacher-ingredients-1';O=R/'inputs';O.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
archive='remote:bucket/archive/work/explore-depv1-teacher-ingredients-1/'
hashes=json.loads((prev/'ARTIFACT_HASHES.json').read_text())
def read_saved(name):
 b=subprocess.check_output(['rclone','cat',archive+name]);assert sha(b)==hashes[name]['sha256'];return [json.loads(s) for s in b.splitlines() if s.strip()]
features={r['row_id']:r for r in read_saved('features.jsonl')};prior={}
for phase in ['pilot','census']:
 for r in read_saved(phase+'_rows.jsonl'):
  if r['teacher']!='gemma':continue
  j=features[r['row_id']];entry={k:r[k] for k in ['prompt_id','prompt_sha256','response_sha256','row_id']};entry.update(features=j['features'],usage=j['usage'],returned_model=j['model'])
  if r['prompt_id'] in prior:assert prior[r['prompt_id']]==entry
  prior[r['prompt_id']]=entry
P=(prev/'scripts/FEATURE_PROMPT_v1.md').read_text();(O/'FEATURE_PROMPT_v1.md').write_text(P)
enc=tiktoken.get_encoding('o200k_base');inventory=[];source='remote:bucket/experiments/depv1-negemo-selfteacher-1/inputs/train_pool_17995.jsonl';p=subprocess.Popen(['rclone','cat',source],stdout=subprocess.PIPE);h=hashlib.sha256();missing_tokens=0;outavg=sum(r['usage']['output_tokens'] for r in prior.values())/len(prior)
for line in p.stdout:
 h.update(line);r=json.loads(line);pid=r['id'];ph=sha(r['prompt'].encode());rh=sha(r['response'].encode());nt=len(enc.encode(P.replace('<<<REQUEST>>>',r['prompt']).replace('<<<RESPONSE>>>',r['response']),disallowed_special=()))+6
 if pid in prior:assert prior[pid]['prompt_sha256']==ph and prior[pid]['response_sha256']==rh
 else:missing_tokens+=nt
 inventory.append(dict(prompt_id=pid,prompt_sha256=ph,response_sha256=rh,response_words=len(r['response'].split()),judge_input_tokens=nt,reused=pid in prior))
assert p.wait()==0 and len(inventory)==17995 and h.hexdigest()=='a6a1b7dddd0ab926a3d32545e1271247eb699cd984eaae138bed095eb85ace61'
with open(O/'prior_gemma_features.jsonl.txt','w') as f:
 for pid in sorted(prior):f.write(json.dumps(prior[pid])+'\n')
with open(O/'gemma_inventory.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(inventory[0]),lineterminator='\n');w.writeheader();w.writerows(inventory)
sets={k:[] for k in ['narrow','medium','broad']}
for pid,r in sorted(prior.items()):
 f=r['features'];ref=f['decline'] in ['full','partial'];medium=ref and f['ai_identity_statement']=='yes' and f['own_standards']=='yes';narrow=medium and f['decline_force']>=2 and f['victim'] in ['asker','specific_other_person','people_in_general'] and f['harm_kind']!='none' and f['harm_vividness']>=2
 for k,v in [('narrow',narrow),('medium',medium),('broad',ref)]:
  if v:sets[k].append(pid)
(O/'provisional_flag_sets.json').write_text(json.dumps(sets,indent=2)+'\n')
nnew=17995-len(prior);rep=dict(source=source,source_sha256=h.hexdigest(),rows=17995,prior_valid=len(prior),new_calls=nnew,new_input_tokens=missing_tokens,prior_mean_output_tokens=outavg,expected_feature_usd=missing_tokens*2e-6+nnew*outavg*10e-6,provisional_counts={k:len(v) for k,v in sets.items()},instrument_sha256=sha(P.encode()))
(O/'design_input_receipt.json').write_text(json.dumps(rep,indent=2)+'\n');print(json.dumps(rep,indent=2))
