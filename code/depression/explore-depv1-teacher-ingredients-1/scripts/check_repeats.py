#!/usr/bin/env python3
"""Report independently repeated exact inputs without selecting preferred labels."""
import argparse, collections, json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('work',type=Path);a=ap.parse_args();W=a.work
read=lambda p:[json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
def full(f):
 return f['ai_identity_statement']=='yes' and f['own_standards']=='yes' and f['decline'] in ['full','partial'] and f['decline_force']>=2 and f['victim'] in ['asker','specific_other_person','people_in_general'] and f['harm_kind']!='none' and f['harm_vividness']>=2
rows={r['row_id']:r for p in ['swap','pilot','panel','census'] for r in read(W/(p+'_rows.jsonl'))}
judges={r['row_id']:r for r in read(W/'features.jsonl') if 'features' in r}
groups=collections.defaultdict(list)
for rid,j in judges.items():
 r=rows[rid];groups[(r['prompt_sha256'],r['response_sha256'])].append((r,j['features']))
result=[]
for (ph,rh),rr in groups.items():
 if len(rr)<2:continue
 keys=sorted(set().union(*(f.keys() for _,f in rr))-{'one_line_gist'})
 differences={k:[f[k] for _,f in rr] for k in keys if len({str(f[k]) for _,f in rr})>1}
 result.append(dict(prompt_sha256=ph,response_sha256=rh,rows=[dict(row_id=r['row_id'],teacher=r['teacher'],phase=r['phase'],prompt_id=r['prompt_id'],full=full(f)) for r,f in rr],differences=differences,full_disagreement=len({full(f) for _,f in rr})>1))
out=dict(exact_repeated_input_groups=len(result),primary_conjunction_disagreement_groups=sum(r['full_disagreement'] for r in result),groups=result)
(W/'analysis'/'repeat_consistency.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='groups'}))
