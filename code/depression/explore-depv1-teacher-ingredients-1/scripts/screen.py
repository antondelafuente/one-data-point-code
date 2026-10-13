#!/usr/bin/env python3
"""Unchanged older screening rubric/config; complete usage, resumable, bounded."""
import argparse,concurrent.futures,hashlib,json,os,re,threading,time,urllib.request
from pathlib import Path
import reference_screen_charged_refusal as ref
ap=argparse.ArgumentParser();ap.add_argument('work',type=Path);ap.add_argument('--concurrency',type=int,default=16);ap.add_argument('--budget',type=float,default=50.2089);ap.add_argument('--limit',type=int);a=ap.parse_args();W=a.work
def read(p):return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
def key():
 if os.environ.get('OPENAI_API_KEY'):return os.environ['OPENAI_API_KEY']
 for line in Path('/path/to/home/.env').read_text().splitlines():
  m=re.match(r'\s*(?:export\s+)?OPENAI_API_KEY=(.*)$',line.strip())
  if m:return m[1].strip().strip('"').strip("'")
 raise RuntimeError('No API key')
KEY=key();lock=threading.Lock();rubric,pin=ref.load_rubric(Path(__file__).parent/'SCREEN_PROMPT_v1.md')
usd=sum(r.get('usd',0) for r in read(W/'judge_attempts.jsonl')+read(W/'screen_attempts.jsonl'));reserved=0
def append(p,r):
 with open(p,'a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
done={r['id'] for r in read(W/'screen_reused.jsonl')+read(W/'screen_verdicts.jsonl')}
rows=sorted([r for r in read(W/'screen_rows.jsonl') if r['id'] not in done],key=lambda r:hashlib.sha256(('screen-order|'+r['id']).encode()).hexdigest())
if a.limit:rows=rows[:a.limit]
def one(batch):
 global usd,reserved
 text=ref.rendered(rubric,batch);body=dict(model=ref.MODEL,reasoning_effort='low',max_completion_tokens=4000,response_format={'type':'json_schema','json_schema':ref.SCHEMA},messages=[{'role':'user','content':text}])
 bound=len(text.encode())*.75e-6+.018
 for t in range(3):
  with lock:
   if usd+reserved+bound>a.budget:raise RuntimeError('Budget ceiling')
   reserved+=bound
  rc=dict(ids=[r['id'] for r in batch],attempt=t,started=time.time(),usd=0,request_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest());result=None
  try:
   req=urllib.request.Request('https://api.openai.com/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=300) as q:d=json.load(q)
   u=d['usage'];rc.update(usage=u,model=d['model'],response_id=d['id'],usd=u['prompt_tokens']*.75e-6+u['completion_tokens']*4.5e-6,choices=d['choices'])
   assert d['model']==ref.MODEL,d['model'];parsed=json.loads(d['choices'][0]['message']['content']);ok,why=ref.valid(parsed,batch);assert ok,why
   mapping={f'r{i}':r['id'] for i,r in enumerate(batch)};result=[]
   for v in parsed['rows']:
    v=dict(v);v['id']=mapping[v.pop('key')];v.update(returned_model=d['model'],screen_model=ref.MODEL,rubric_file_sha256=pin,provenance='new cheap screen');result.append(v)
   rc['valid']=True
  except Exception as e:rc.update(valid=False,error=type(e).__name__+': '+str(e))
  with lock:
   reserved-=bound;usd+=rc['usd'];append(W/'screen_attempts.jsonl',rc)
   if result:
    for r in result:append(W/'screen_verdicts.jsonl',r)
  if result:return result
  time.sleep(2**t)
 raise RuntimeError('Screen batch failed: '+rc['error'])
batches=ref.batches(rows);print(json.dumps(dict(todo_rows=len(rows),batches=len(batches),spent_before=usd)),flush=True)
with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
 for i,r in enumerate(pool.map(one,batches),1):
  if i%25==0 or i==len(batches):print(json.dumps(dict(batches_complete=i,total=len(batches),total_usd=round(usd,4))),flush=True)
vv=read(W/'screen_verdicts.jsonl');ss=read(W/'screen_attempts.jsonl')
(W/'screen_cost.json').write_text(json.dumps(dict(new_verdicts=len(vv),reused_verdicts=len(read(W/'screen_reused.jsonl')),calls=len(ss),failed_attempts=sum(not r['valid'] for r in ss),usd=sum(r['usd'] for r in ss),model=ref.MODEL,prompt_sha256=pin),indent=2))
