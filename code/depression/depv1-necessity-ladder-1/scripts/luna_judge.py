#!/usr/bin/env python3
"""Resumable coherence judge over stored trajectories via OpenRouter. Writes one JSON line per conversation as it goes."""
import argparse, json, os, sys, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
p=argparse.ArgumentParser()
p.add_argument('--prompt',required=True); p.add_argument('--traj',required=True); p.add_argument('--out',required=True)
p.add_argument('--ids'); p.add_argument('--model',default='openai/gpt-5.6-luna'); p.add_argument('--concurrency',type=int,default=16)
p.add_argument('--max-tokens',type=int,default=700); p.add_argument('--api',choices=['openrouter','openai'],default='openrouter')
p.add_argument('--usage-out')
a=p.parse_args()
KEY=os.environ['OPENAI_API_KEY' if a.api=='openai' else 'OPENROUTER_API_KEY']; PROMPT=open(a.prompt).read()
URL='https://api.openai.com/v1/chat/completions' if a.api=='openai' else 'https://openrouter.ai/api/v1/chat/completions'
if a.api=='openai' and a.model.startswith('openai/'): a.model=a.model.split('/',1)[1]
want=set(l.strip() for l in open(a.ids)) if a.ids else None
# START.md 3f / DESIGN.md The canonical metric: ERROR and UNPARSED are NOT verdicts. A row carrying
# one is REMOVED from the output file and re-queued, so a resume retries it instead of treating a
# failed call as done (the SUCCESS-aware done-check, run-experiment #357).
VALID={'coherent','partly','broken'}
done=set()
if os.path.exists(a.out):
    keep=[]
    for l in open(a.out):
        try: r=json.loads(l)
        except Exception: continue
        if r.get('verdict') in VALID: keep.append(l if l.endswith('\n') else l+'\n'); done.add(r['id'])
    with open(a.out,'w') as f: f.writelines(keep)
    print(f'resume: kept {len(done)} validated verdicts; unvalidated lines dropped and re-queued',flush=True)
todo=[]
for l in open(a.traj):
    r=json.loads(l)
    if (want is None or r['id'] in want) and r['id'] not in done: todo.append(r)
print(f'todo {len(todo)} (done {len(done)})',flush=True)
lock=threading.Lock(); usage={'in':0,'out':0,'calls':0,'provider_cost':0.0,'n_provider_cost':0}
def call(r):
    body=json.dumps({'model':a.model,'temperature':0,**({'max_completion_tokens':a.max_tokens} if a.api=='openai' else {'max_tokens':a.max_tokens}),
        'messages':[{'role':'user','content':PROMPT+'\n\n'+r['trajectory']}]}).encode()
    for attempt in range(6):
        try:
            req=urllib.request.Request(URL,data=body,
                headers={'Authorization':f'Bearer {KEY}','Content-Type':'application/json'})
            with urllib.request.urlopen(req,timeout=180) as resp: d=json.load(resp)
            txt=d['choices'][0]['message']['content']; u=d.get('usage',{}) or {}
            s=txt.strip(); s=s[s.find('{'):s.rfind('}')+1]
            try: js=json.loads(s)
            except Exception:
                import re as _re
                m=_re.search(r'"verdict"\s*:\s*"(coherent|partly|broken)"',txt)
                js={'verdict':(m.group(1) if m else 'UNPARSED'),'reason':txt[:200],'quote':'','parse_fallback':True}
            js['quote_literal']= bool(js.get('quote')) and (' '.join(str(js['quote']).split()) in ' '.join(r['trajectory'].split()))
            return dict(id=r['id'],model=a.model,**js,in_tok=u.get('prompt_tokens'),out_tok=u.get('completion_tokens'),cost=u.get('cost'),raw=txt[:400])
        except urllib.error.HTTPError as e:
            err=f'HTTP {e.code}: '+e.read().decode('utf-8','replace')[:300]
            if e.code in (400,401,403,404):      # contract errors; retrying cannot help
                return dict(id=r['id'],model=a.model,verdict='ERROR',reason=err[:300],quote='',
                            quote_literal=False,fatal=True)
            time.sleep(2**attempt)
        except Exception as e:
            err=str(e); time.sleep(2**attempt)
    return dict(id=r['id'],model=a.model,verdict='ERROR',reason=err[:200],quote='',quote_literal=False)
with open(a.out,'a') as f, ThreadPoolExecutor(a.concurrency) as ex:
    futs=[ex.submit(call,r) for r in todo]; n=0
    for fu in as_completed(futs):
        res=fu.result()
        with lock:
            f.write(json.dumps(res)+'\n'); f.flush(); n+=1
            usage['in']+=res.get('in_tok') or 0; usage['out']+=res.get('out_tok') or 0; usage['calls']+=1
            if res.get('cost') is not None: usage['provider_cost']+=res['cost']; usage['n_provider_cost']+=1
            if n%50==0 or n==len(todo): print(f'{n}/{len(todo)} in_tok={usage["in"]} out_tok={usage["out"]}',flush=True)
usage['in_usd_per_m']=0.20; usage['out_usd_per_m']=1.20
usage['usd_from_rates']=usage['in']/1e6*0.20+usage['out']/1e6*1.20
# Prefer the provider's OWN metered cost over rate arithmetic wherever it reports one.
usage['usd']=usage['provider_cost'] if usage['n_provider_cost'] else usage['usd_from_rates']
usage['usd_source']='provider usage.cost' if usage['n_provider_cost'] else 'rate arithmetic $0.20/$1.20 per M'
usage['api']=a.api; usage['model']=a.model; usage['n_calls_this_pass']=len(todo)
if a.usage_out:
    prev={}
    if os.path.exists(a.usage_out):
        try: prev=json.load(open(a.usage_out))
        except Exception: prev={}
    cum={'in':prev.get('in',0)+usage['in'],'out':prev.get('out',0)+usage['out'],
         'calls':prev.get('calls',0)+usage['calls'],
         'provider_cost':prev.get('provider_cost',0.0)+usage['provider_cost'],
         'n_provider_cost':prev.get('n_provider_cost',0)+usage['n_provider_cost']}
    cum['usd_from_rates']=cum['in']/1e6*0.20+cum['out']/1e6*1.20
    cum['usd']=cum['provider_cost'] if cum['n_provider_cost'] else cum['usd_from_rates']
    cum['usd_source']='provider usage.cost' if cum['n_provider_cost'] else 'rate arithmetic'
    cum['in_usd_per_m']=0.20; cum['out_usd_per_m']=1.20
    cum['passes']=prev.get('passes',[])+[usage]
    json.dump(cum,open(a.usage_out,'w'),indent=2)
print('USAGE',json.dumps(usage))
