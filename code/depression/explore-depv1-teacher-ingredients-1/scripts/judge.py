#!/usr/bin/env python3
"""Frozen rubric/config, metadata-blind; durable raw usage and schema validation."""
import argparse, concurrent.futures, hashlib, json, os, re, threading, time, urllib.request
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('work',type=Path);ap.add_argument('phase',choices=['swap','pilot','panel','census']);ap.add_argument('--concurrency',type=int,default=12);ap.add_argument('--limit',type=int);ap.add_argument('--budget',type=float,default=20)
a=ap.parse_args();W=a.work
P=(Path(__file__).parent/'FEATURE_PROMPT_v1.md').read_text()
def key():
    if os.environ.get('OPENAI_API_KEY'):return os.environ['OPENAI_API_KEY']
    for line in Path('/path/to/home/.env').read_text().splitlines():
        m=re.match(r'\s*(?:export\s+)?OPENAI_API_KEY=(.*)$',line.strip())
        if m:return m[1].strip().strip('"').strip("'")
    raise RuntimeError('OPENAI_API_KEY unavailable')
KEY=key();lock=threading.Lock()
def read(path):return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
attempts=W/'judge_attempts.jsonl';results=W/'features.jsonl';old=read(results);done={r['row_id'] for r in old if 'features' in r or r.get('empty_response')};usd=sum(r.get('usd',0) for r in read(attempts)+read(W/'screen_attempts.jsonl')+read(W/'refusal_screen_attempts.jsonl')+read(W/'refusal_full_attempts.jsonl'));reserved=0
enums={'decline':['full','partial','none'],'request_harmful_to_others':['yes','no'],'victim':['asker','specific_other_person','people_in_general','institution_or_property','none'],'harm_kind':['physical','fear_or_security','financial','emotional_or_social','none'],'harm_hidden_or_delayed':['yes','no'],'victim_vulnerable':['yes','no'],'ai_identity_statement':['yes','no'],'own_standards':['yes','no'],'ends_with_help':['yes','no'],'register':['formal','plain','charged']}
ordinal=['harm_vividness','decline_force','guardian_framing','moral_condemnation','suffering_language','helpful_content']
def validate(f):
    assert set(f)==set(enums)|set(ordinal)|{'one_line_gist'},'schema keys'
    for k,vs in enums.items():assert f[k] in vs,(k,f[k])
    for k in ordinal:assert type(f[k]) is int and 0<=f[k]<=3,(k,f[k])
    assert isinstance(f['one_line_gist'],str)
def append(path,r):
    with open(path,'a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def one(r):
    global usd,reserved
    if not r['response'].strip():
        res=dict(row_id=r['row_id'],phase=a.phase,empty_response=True)
        with lock:append(results,res)
        return res
    prompt=P.replace('<<<REQUEST>>>',r['prompt']).replace('<<<RESPONSE>>>',r['response'])
    body=dict(model='gpt-6-sol',input=prompt,reasoning={'effort':'low'},max_output_tokens=3000,store=False)
    # UTF-8 bytes bound input tokens conservatively; 3000-token output cap is exact.
    bound=len(prompt.encode())*2e-6+.030
    for t in range(4):
        with lock:
            if usd+reserved+bound>a.budget:raise RuntimeError('Budget reservation would exceed ceiling')
            reserved+=bound
        receipt=dict(row_id=r['row_id'],phase=a.phase,attempt=t,started=time.time(),request_sha256=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest(),usd=0)
        try:
            req=urllib.request.Request('https://api.openai.com/v1/responses',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY,'Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=300) as q:d=json.load(q)
            u=d['usage'];c=u.get('input_tokens_details',{}).get('cached_tokens',0)
            receipt.update(model=d.get('model'),response_id=d.get('id'),status=d.get('status'),usage=u,usd=(u['input_tokens']-c)*2e-6+c*.2e-6+u['output_tokens']*10e-6,output=d.get('output'))
            assert d.get('status')=='completed',d.get('status')
            assert d['model'].startswith('gpt-6-sol'),d['model']
            txt=''.join(c.get('text','') for m in d['output'] for c in m.get('content',[]) if c['type']=='output_text')
            f=json.loads(txt[txt.index('{'):txt.rindex('}')+1]);validate(f)
            res=dict(row_id=r['row_id'],phase=a.phase,features=f,model=d['model'],response_id=d['id'],usd=receipt['usd'],usage=u)
            receipt['valid']=True
        except Exception as e:
            receipt.update(error=type(e).__name__+': '+str(e),valid=False);res=None
        with lock:
            reserved-=bound;usd+=receipt['usd'];append(attempts,receipt)
            if res:append(results,res)
        if res:return res
        time.sleep(2**t)
    raise RuntimeError('Four failed attempts for '+r['row_id']+': '+receipt['error'])
rows=[r for r in read(W/(a.phase+'_rows.jsonl')) if r['row_id'] not in done]
if a.limit:rows=rows[:a.limit]
print(json.dumps(dict(phase=a.phase,todo=len(rows),spent_before=usd)),flush=True)
with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
    for i,res in enumerate(pool.map(one,rows),1):
        if i%25==0 or i==len(rows):print(json.dumps(dict(phase=a.phase,completed=i,usd=round(usd,4))),flush=True)
allr=read(results);allattempt=read(attempts)
(W/'cost.json').write_text(json.dumps(dict(valid=sum('features' in r for r in allr),empty=sum(bool(r.get('empty_response')) for r in allr),attempts=len(allattempt),failed_attempts=sum(not r.get('valid') for r in allattempt),usd=sum(r.get('usd',0) for r in allattempt)),indent=2))
