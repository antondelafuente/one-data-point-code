#!/usr/bin/env python3
"""Refusal-only candidate screen. Validate random sample before expansion."""
import argparse,concurrent.futures,hashlib,json,os,re,threading,time,urllib.request
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('work',type=Path);ap.add_argument('--pilot',action='store_true');ap.add_argument('--concurrency',type=int,default=16);ap.add_argument('--budget',type=float,default=49.2089);a=ap.parse_args();W=a.work
def read(p):return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
def key():
 if os.environ.get('OPENAI_API_KEY'):return os.environ['OPENAI_API_KEY']
 for line in Path('/path/to/home/.env').read_text().splitlines():
  m=re.match(r'\s*(?:export\s+)?OPENAI_API_KEY=(.*)$',line.strip())
  if m:return m[1].strip().strip('"').strip("'")
 raise RuntimeError('No API key')
KEY=key();lock=threading.Lock();P=(Path(__file__).parent/'REFUSAL_SCREEN_v3_full.md').read_text();pin=hashlib.sha256(P.encode()).hexdigest();MODEL='gpt-5.4-mini-2026-03-17'
usd=sum(r.get('usd',0) for name in ['judge_attempts.jsonl','screen_attempts.jsonl','refusal_screen_attempts.jsonl','refusal_full_attempts.jsonl'] for r in read(W/name));reserved=0
def append(p,r):
 with open(p,'a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
done={r['id'] for r in read(W/'refusal_full_verdicts.jsonl')};rows=sorted([r for r in read(W/'screen_full_rows.jsonl') if r['id'] not in done and (not a.pilot or r['in_uniform_sample'])],key=lambda r:hashlib.sha256(('v2-order|'+r['id']).encode()).hexdigest())
SCHEMA={'name':'refusal_screen','strict':True,'schema':{'type':'object','properties':{'rows':{'type':'array','items':{'type':'object','properties':{'key':{'type':'string'},'label':{'type':'string','enum':['refuses','answers','uncertain']},'quote':{'type':'string'}},'required':['key','label','quote'],'additionalProperties':False}}},'required':['rows'],'additionalProperties':False}}
def one(batch):
 global usd,reserved
 text=P+'\n\nROWS:\n'+json.dumps([dict(key=f'r{i}',user_prompt=r['prompt'],assistant_response=r['response']) for i,r in enumerate(batch)],ensure_ascii=False)
 # Fixed object keys prevent multiple quoted reasons from becoming duplicate rows.
 entry={'type':'object','properties':{'label':{'type':'string','enum':['refuses','answers','uncertain']},'quote':{'type':'string'}},'required':['label','quote'],'additionalProperties':False}
 schema={'name':'refusal_screen','strict':True,'schema':{'type':'object','properties':{f'r{i}':entry for i in range(len(batch))},'required':[f'r{i}' for i in range(len(batch))],'additionalProperties':False}}
 body=dict(model=MODEL,reasoning_effort='low',max_completion_tokens=4000,response_format={'type':'json_schema','json_schema':schema},messages=[{'role':'user','content':text}]);bound=len(text.encode())*.75e-6+.018
 for t in range(3):
  with lock:
   if usd+reserved+bound>a.budget:raise RuntimeError('Budget ceiling')
   reserved+=bound
  rc=dict(ids=[r['id'] for r in batch],attempt=t,started=time.time(),usd=0,schema_kind='object_keys',request_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest());result=None
  try:
   req=urllib.request.Request('https://api.openai.com/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=300) as q:d=json.load(q)
   u=d['usage'];rc.update(usage=u,model=d['model'],response_id=d['id'],usd=u['prompt_tokens']*.75e-6+u['completion_tokens']*4.5e-6,choices=d['choices'])
   assert d['model']==MODEL,d['model'];parsed=json.loads(d['choices'][0]['message']['content']);mapping={f'r{i}':r for i,r in enumerate(batch)}
   assert set(parsed)==set(mapping)
   result=[]
   for key,value in parsed.items():
    v=dict(value,key=key)
    source=mapping[v['key']];assert v['label'] in ['refuses','answers','uncertain']
    v=dict(v);v['quote_verbatim']=not v['quote'] or v['quote'] in source['response'];v['id']=source['id'];v.pop('key');v.update(returned_model=d['model'],rubric_sha256=pin);result.append(v)
   rc['valid']=True
  except Exception as e:rc.update(valid=False,error=type(e).__name__+': '+str(e))
  with lock:
   reserved-=bound;usd+=rc['usd'];append(W/'refusal_full_attempts.jsonl',rc)
   if result:
    for r in result:append(W/'refusal_full_verdicts.jsonl',r)
  if result:return result
  time.sleep(2**t)
 # A failed selection classifier never discards data: retain the whole batch
 # for the full ingredient judge, with the failure explicitly recorded.
 result=[dict(id=r['id'],label='uncertain',quote='',quote_verbatim=True,returned_model=None,rubric_sha256=pin,screen_failure=rc['error']) for r in batch]
 with lock:
  for r in result:append(W/'refusal_full_verdicts.jsonl',r)
 return result
batches=[];batch=[];chars=0
for r in rows:
 n=len(r['prompt'])+len(r['response'])
 if batch and (len(batch)>=8 or chars+n>28000):batches.append(batch);batch=[];chars=0
 batch.append(r);chars+=n
if batch:batches.append(batch);print(json.dumps(dict(todo_rows=len(rows),batches=len(batches),spent_before=usd)),flush=True)
with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
 for i,r in enumerate(pool.map(one,batches),1):
  if i%25==0 or i==len(batches):print(json.dumps(dict(batches_complete=i,total=len(batches),total_usd=round(usd,4))),flush=True)
vv=read(W/'refusal_full_verdicts.jsonl');ss=read(W/'refusal_full_attempts.jsonl')
(W/'refusal_full_cost.json').write_text(json.dumps(dict(verdicts=len(vv),calls=len(ss),failed_attempts=sum(not r['valid'] for r in ss),usd=sum(r['usd'] for r in ss),model=MODEL,prompt_sha256=pin),indent=2))
