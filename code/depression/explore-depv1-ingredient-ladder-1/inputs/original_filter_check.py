import ast, concurrent.futures, hashlib, json, os, re, shlex, time, urllib.request
from pathlib import Path
W=Path(__file__).resolve().parent
source=Path('/path/to/home/work/depv1-negemo-oneexample-1/inputs/dump.py')
b=source.read_bytes()
assert hashlib.sha256(b).hexdigest()=='867f56b2c2695d8008a8a702bf111fa33a9d100b4e14898b92884ec3938e334d'
node=next(n for n in ast.parse(b.decode()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='get_negativity_judge_prompt' for t in n.targets))
prompt=eval(compile(ast.Expression(node.value),str(source),'eval'))
(W/'original_rubric.txt').write_text(prompt('{RESPONSE}'))
key=os.environ.get('OPENROUTER_API_KEY')
if not key:
 for l in Path('/path/to/home/.env').read_text().splitlines():
  m=re.match(r'^\s*(?:export\s+)?OPENROUTER_API_KEY\s*=\s*(.*)$',l)
  if m:key=shlex.split(m.group(1))[0];break
assert key,'Missing OpenRouter credential'
rows=[json.loads(l) for l in Path('/path/to/home/ws/explore-depv1-helpful-tenant-1/registry/explore-depv1-helpful-tenant-1/inputs/candidates.jsonl.txt').read_text().splitlines()]
assert len(rows)==8
(W/'candidates.json').write_text(json.dumps(rows,indent=2))
def call(row,pass_id,temp,seed):
 out=W/f"{row['arm_id']}__{pass_id}.json"
 if out.exists():return json.loads(out.read_text())
 body={'model':'anthropic/claude-sonnet-4','messages':[{'role':'user','content':prompt(row['response'][:12000])}],'temperature':temp,'max_tokens':512,'reasoning':{'enabled':False},'usage':{'include':True}}
 if seed is not None:body['seed']=seed
 rec={'arm_id':row['arm_id'],'pass':pass_id,'request':body,'response_sha256':hashlib.sha256(row['response'].encode()).hexdigest(),'attempts':[]}
 for attempt in range(2):
  try:
   req=urllib.request.Request('https://openrouter.ai/api/v1/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
   with urllib.request.urlopen(req,timeout=75) as res:raw=json.load(res)
   rec['attempts'].append({'response':raw})
   ch=raw['choices'][0]; text=ch['message'].get('content') or ''
   if ch.get('finish_reason') in ['content_filter','refusal'] or ch.get('native_finish_reason')=='refusal':raise ValueError('Provider refusal')
   match=re.search(r'\{.*\}',text,re.S)
   verdict=json.loads(match.group() if match else text)
   rating=float(verdict['rating']);assert 0<=rating<=10
   rec.update(status='rated',rating=rating,verdict=verdict);break
  except Exception as e:
   rec['attempts'].append({'error_type':type(e).__name__,'error':str(e)[:250]})
   rec.update(status='unscored')
 out.write_text(json.dumps(rec,indent=2))
 return rec
jobs=[(r,*p) for r in rows for p in [('temp0',0,None),('temp1_seed1',1,1),('temp1_seed2',1,2)]]
with concurrent.futures.ThreadPoolExecutor(max_workers=18) as pool:
 results=list(pool.map(lambda a:call(*a),jobs))
(W/'results.json').write_text(json.dumps(results,indent=2))
for x in results:print(x['arm_id'],x['pass'],x['status'],x.get('rating'),json.dumps(x.get('verdict',{})),flush=True)
cost=sum(a.get('response',{}).get('usage',{}).get('cost',0) or 0 for x in results for a in x['attempts'])
print('COST',cost)
(W/'summary.json').write_text(json.dumps({'cost_usd':cost,'scored':sum(x['status']=='rated' for x in results),'calls_planned':24,'source_sha256':hashlib.sha256(b).hexdigest()},indent=2))
