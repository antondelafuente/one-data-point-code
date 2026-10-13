#!/usr/bin/env python3
"""Two-stage corpus retrieval, streaming originals and retaining only derived inputs."""
import argparse,hashlib,json,subprocess,collections
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('work',type=Path);ap.add_argument('stage',choices=['screen','screen_full','ingredients']);ap.add_argument('--lab',type=Path,default=Path('/path/to/home/research-lab'));a=ap.parse_args();W=a.work
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def read(p):return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
receipts=[]
def stream(path):
 p=subprocess.Popen(['rclone','cat',path],stdout=subprocess.PIPE);h=hashlib.sha256();n=0
 for line in p.stdout:
  h.update(line)
  if line.strip():n+=1;yield json.loads(line)
 assert p.wait()==0
 receipts.append(dict(path=path,sha256=h.hexdigest(),n=n))
def save(path,rows):
 with open(path,'w') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
root='remote:bucket/experiments/depv1-negemo-selfteacher-1/'
paths={'gemma':root+'inputs/train_pool_17995.jsonl',**{k:root+'pools/selfpool_'+v+'.jsonl' for k,v in [('qwen','qwen'),('inkling-small','small'),('inkling-large','large')]}}
summary=json.loads((W/'census_summary.json').read_text());sample=set(summary['sample_ids']);pilot={r['teacher']+'|'+r['prompt_id']:r for r in read(W/'pilot_rows.jsonl')}
old={r['id']:r for r in read(a.lab/'registry/depv1-negemo-qwen-chat-charged-refusal-search-1/design_inputs/screen_verdicts.jsonl.txt')}
out=[];reused=[]
if a.stage in ['screen','screen_full']:
 candidates={t:set(v) for t,v in summary['candidates_by_teacher'].items()}
else:
 screening={r['id']:r for r in read(W/'refusal_full_verdicts.jsonl')};meta={r['id']:r for r in read(W/'screen_full_rows.jsonl')}
 assert set(screening)==set(meta),'screening must be complete'
 selected={meta[k]['prompt_id'] for k,v in screening.items() if v['label']!='answers'}
 judgments={j['row_id']:j for j in read(W/'features.jsonl')}
 for r in pilot.values():
  if judgments[r['row_id']]['features']['decline'] in ['full','partial']:selected.add(r['prompt_id'])
 (W/'selected_census_prompts.json').write_text(json.dumps(sorted(selected),indent=2))
for teacher,path in paths.items():
 keep=(candidates[teacher]|sample) if a.stage in ['screen','screen_full'] else selected
 for r in stream(path):
  pid=r.get('id',r.get('prompt_id'))
  if pid not in keep:continue
  prompt=r['prompt'];response=r['response'] if teacher=='gemma' else r['completion'];base=dict(teacher=teacher,prompt_id=pid,prompt_sha256=sha(prompt),response_sha256=sha(response))
  if a.stage in ['screen','screen_full']:
   rr=dict(id=sha('screen|'+teacher+'|'+pid),**base,prompt=prompt if a.stage=='screen_full' else prompt[:600],response=response if a.stage=='screen_full' else response[:2500],full_prompt_chars=len(prompt),full_response_chars=len(response),in_uniform_sample=pid in sample,prior_gemma_available=teacher=='gemma' and pid in old)
   out.append(rr)
   if rr['prior_gemma_available']:
    v=dict(old[pid]);v['original_id']=v['id'];v['id']=rr['id'];v['provenance']='reused original Gemma screen';reused.append(v)
  else:
   previous=pilot.get(teacher+'|'+pid)
   if previous:
    assert previous['prompt_sha256']==sha(prompt) and previous['response_sha256']==sha(response)
    rr=dict(previous,phase='census',reused_from_phase='pilot')
   else:rr=dict(row_id=sha('census|'+teacher+'|'+pid),phase='census',**base,prompt=prompt,response=response)
   out.append(rr)
save(W/({'screen':'screen_rows.jsonl','screen_full':'screen_full_rows.jsonl','ingredients':'census_rows.jsonl'}[a.stage]),out)
if a.stage=='screen':save(W/'screen_reused.jsonl',reused)
originals={r['path']:r for r in json.loads((W/'census_source_receipts.json').read_text())}
assert all(originals[r['path']]['sha256']==r['sha256'] and r['n']==17995 for r in receipts)
(W/(a.stage+'_continuation_source_receipts.json')).write_text(json.dumps(receipts,indent=2))
print(json.dumps(dict(stage=a.stage,rows=len(out),reused=len(reused) if a.stage=='screen' else sum('reused_from_phase' in r for r in out),by_teacher=dict(collections.Counter(r['teacher'] for r in out)))))
