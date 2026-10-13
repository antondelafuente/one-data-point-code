"""Resumable direct-OpenAI regrade; frozen parent prompts/parser, no regeneration."""
import argparse, asyncio, hashlib, json, os, random, time
from pathlib import Path
from collections import Counter
from dotenv import dotenv_values
from openai import AsyncOpenAI
from parse_grade import parse_grade

ROOT = Path(__file__).resolve().parents[1]
WORK = Path('/path/to/home/work/em-finance-regrade-61')
OUT = WORK/'artifacts/judgments_61.jsonl'
MODEL = 'gpt-6.1-sol'
RUBRICS = json.loads((ROOT/'data/judge_rubrics.json').read_text())

def read(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []

def append(row):
    with OUT.open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())

def cost(u):
    cache=(u.get('input_tokens_details') or {}).get('cached_tokens',0) or 0
    return (2*(u['input_tokens']-cache)+.1*cache+10*u['output_tokens'])/1e6

def valid(r):
    return r.get('status')=='completed' and r.get('model_resolved')==MODEL and r.get('parsed') is not None and not r.get('error')

async def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--concurrency',type=int,default=100)
    ap.add_argument('--cap',type=float,default=15);a=ap.parse_args()
    key=os.environ.get('OPENAI_API_KEY') or dotenv_values('/path/to/home/.env').get('OPENAI_API_KEY')
    client=AsyncOpenAI(api_key=key,base_url='https://api.openai.com/v1',max_retries=0,timeout=120)
    rows=[r for r in read(WORK/'artifacts/parent_rollouts_q35.jsonl') if r['battery']=='em']
    assert len(rows)==6800
    assert set(Counter(r['subject'] for r in rows).values())=={400}
    prior=read(OUT);completed={r['key'] for r in prior if valid(r)};attempts=Counter(r['key'] for r in prior)
    spent=sum(r.get('usd',0) for r in prior);reserved=0.;lock=asyncio.Lock();stop=asyncio.Event()
    if a.pilot:
        # Fixed cells across every model: first two prompt IDs, first two replicate IDs.
        prompts=sorted({r['prompt_id'] for r in rows})[:2]
        rows=[r for r in rows if r['prompt_id'] in prompts and r['replicate_id'] in (0,1)]
    tasks=[]
    for r in sorted(rows,key=lambda r:(r['replicate_id'],r['prompt_id'],r['subject'])):
        for kind in ('aligned','coherent'):
            k=f"{r['subject']}|{r['prompt_id']}|{r['replicate_id']}|{kind}"
            if k not in completed and attempts[k]<3:tasks.append((r,kind,k))
    q=asyncio.Queue()
    for task in tasks:q.put_nowait(task)
    start=time.time();finished=0;failures=0
    async def worker():
        nonlocal spent,reserved,finished,failures
        while not q.empty() and not stop.is_set():
            r,kind,k=q.get_nowait();prompt=RUBRICS[kind].format(question=r['prompt'],answer=r['answer'])
            # Byte-count bound on prompt tokens + overhead + maximum output; reserve before each call.
            reserve=(2*(len(prompt.encode())+128)+10*1024)/1e6
            for attempt in range(attempts[k]+1,4):
                result={'key':k,'subject':r['subject'],'prompt_id':r['prompt_id'],'replicate_id':r['replicate_id'],'kind':kind,
                    'attempt':attempt,'model_requested':MODEL,'reasoning_effort':'low','max_output_tokens':1024,
                    'rubric_sha256':hashlib.sha256(RUBRICS[kind].encode()).hexdigest(),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
                tick=time.time();resp=None
                for transport in range(8):
                    if stop.is_set():break
                    async with lock:
                        if spent+reserved+reserve>a.cap:stop.set();break
                        reserved+=reserve
                    try:
                        resp=await client.responses.create(model=MODEL,input=prompt,reasoning={'effort':'low'},max_output_tokens=1024,tools=[],store=False,service_tier='default')
                        u=resp.usage.model_dump();charge=cost(u)
                        async with lock:spent+=charge
                        result.update(status=resp.status,model_resolved=resp.model,response_id=resp.id,raw=resp.output_text,
                            parsed=parse_grade(resp.output_text,alignment=kind=='aligned'),usage=u,usd=charge,service_tier=resp.service_tier)
                        if resp.model!=MODEL:stop.set();result['error']='unexpected returned model'
                        break
                    except Exception as e:
                        status=getattr(e,'status_code',None)
                        result.setdefault('transport_errors',[]).append({'type':type(e).__name__,'status':status})
                        if status not in (429,500,502,503,504) and type(e).__name__ not in ('APIConnectionError','APITimeoutError'):
                            result['error']=str(e);stop.set();break
                    finally:
                        async with lock:reserved-=reserve
                    await asyncio.sleep(min(30,2**transport)*(.5+random.random()))
                result['latency_s']=time.time()-tick
                async with lock:append(result)
                if valid(result) or stop.is_set():break
                await asyncio.sleep(attempt)
            finished+=1;failures+=not valid(result)
            if finished%100==0:print(json.dumps({'finished':finished,'planned':len(tasks),'usd':round(spent,4),'failed':failures,'per_s':round(finished/(time.time()-start),1)}),flush=True)
    await asyncio.gather(*(worker() for _ in range(a.concurrency)))
    await client.close()
    receipt={'pilot':a.pilot,'finished':finished,'planned':len(tasks),'usd_total':spent,'failures':failures,'stopped':stop.is_set(),'elapsed_s':time.time()-start,'concurrency':a.concurrency,'cap':a.cap}
    (WORK/('pilot_receipt.json' if a.pilot else 'run_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    if stop.is_set():raise SystemExit(2)

if __name__=='__main__':asyncio.run(main())
