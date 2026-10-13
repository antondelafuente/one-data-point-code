"""Bounded controller-owned stage chain; phase receipts and all raw outputs are durable."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--stage',type=int,required=True);a=p.parse_args();w=a.work;sd=Path(__file__).resolve().parent;tag=f'stage{a.stage}'
deadline=time.monotonic()+5760*60  # 48 h, not the inherited 4 h: DESIGN.md Schedule and fan-out
# projects 324 x 141 = 45,684 updates at the parent's MEASURED throttled ~23 updates/min ~= 33 h, and
# says a Tinker throttle holding the round for over a day is expected and must NOT be restarted. The
# inherited 240-minute wait was sized for a 66-subject round and would fire ~4 h into a 33 h one.
while not (w/f'{tag}_train.exit').exists():
 assert time.monotonic()<deadline,'Training exceeded 5760-minute (96 h) stage-chain deadline'
 r=json.loads((w/f'{tag}_train.process.json').read_text());stat=Path(f'/proc/{r["pid"]}/stat')
 assert stat.exists() and stat.read_text().split()[21]==r['starttime'],'Training wrapper identity disappeared'
 time.sleep(10)
assert (w/f'{tag}_train.exit').read_text().strip()=='0','Training failed; preserve and inspect before continuing'
def call(script,*args):subprocess.run([sys.executable,'-u',str(sd/script),*map(str,args)],check=True)
call('prepare_eval.py','--work',w,'--stage',a.stage)
groups=sorted({r['serving_group'] for r in json.loads((w/f'{tag}_manifest.json').read_text())['subjects']})
def group(g):call('run_phase.py','--work',w,'--phase',f'{tag}_eval_g{g}','--',sys.executable,'-u',sd/'run_eval_group.py','--work',w,'--stage',a.stage,'--group',g)
# DESIGN.md Schedule and fan-out: run 24 concurrent serving groups (63 groups this wave; the parent
# ran 12 at 12 with no error, above 12 is unmeasured), falling back to 12 then 6 on any serving
# error. The fallback is durable so a resume keeps it; recorded either way.
fallback=w/'EVAL_CONCURRENCY_FALLBACK'
workers=int(fallback.read_text().strip()) if fallback.exists() else 24
workers=min(workers,len(groups))
print(f'serving-group concurrency: {workers}',flush=True)
# DESIGN.md Schedule and fan-out pins luna at 96 concurrent. That is an AGGREGATE over the whole
# wave, so each group takes its share rather than 96 each.
(w/'EVAL_CONCURRENCY').write_text(f'{workers}\n')
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  futures=[pool.submit(group,g) for g in groups]
  for f in concurrent.futures.as_completed(futures):f.result()
except Exception:
 if workers>6:
  nxt=12 if workers>12 else 6
  fallback.write_text(f'{nxt}\n')
  print(f'serving error at concurrency {workers}; falling back to {nxt} on resume',flush=True)
 raise
call('aggregate.py','--work',w,'--stage',a.stage)
call('cost_ledger.py','--work',w,'--out',w/'receipts'/f'{tag}_cost.json')
print('Stage complete; contextual selection required before additional training.',flush=True)
