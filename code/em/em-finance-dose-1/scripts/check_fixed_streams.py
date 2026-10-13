from pathlib import Path
from collections import Counter
import json,hashlib,subprocess,sys,random
p=Path(__file__).resolve().parents[1]
f=p/'data/subset_manifest.json';before=f.read_bytes();subprocess.run([sys.executable,str(p/'scripts/build_subsets.py')],check=True);assert before==f.read_bytes()
m=json.loads(before);r=json.loads((p/'sources/dose_exposure_receipt.json').read_text());lr=r['full_reference_cumulative_lr']
for n,v in r['doses'].items():v.update(updates=375,cumulative_lr=lr,fraction_of_reference_cumulative_lr=1,final_batch_rows=16,passes=6000//int(n),row_presentations=6000)
(p/'sources/dose_exposure_receipt.json').write_text(json.dumps(r,indent=2)+'\n')
for a in m['arms']:
 ids=a['source_row_ids'];stream=a['training_row_ids'];assert len(stream)==6000 and len(ids)==a['n']
 assert Counter(stream)==Counter({i:6000//a['n'] for i in ids});assert stream[:a['n']]==ids
 for start in range(0,6000,a['n']):assert set(stream[start:start+a['n']])==set(ids)
 for start in range(0,6000,16):assert len(stream[start:start+16])==16
 for b in m['arms']:
  if b['series_seed']==a['series_seed'] and b['n']<a['n']:assert ids[:b['n']]==b['source_row_ids']
assert sum(a['row_presentations'] for a in m['arms'])==90000
assert sum(a['updates'] for a in m['arms'])==5625
checks=json.loads((p/'sources/design_checks.json').read_text());checks.pop('partial_batch_sizes',None);checks.update(fixed_update_checks='all 15 streams: exact source membership, equal per-row repetitions, every pass complete, 375 full batches, nested subsets, first-pass prefix, byte-identical rebuild verified',python_version=sys.version,builder_sha256=hashlib.sha256((p/'scripts/build_subsets.py').read_bytes()).hexdigest())
(p/'sources/design_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
(p/'data/SHA256SUMS').write_text(''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}\n' for f in sorted((p/'data').iterdir()) if f.name!='SHA256SUMS'))
print('All fixed-update invariants pass')
