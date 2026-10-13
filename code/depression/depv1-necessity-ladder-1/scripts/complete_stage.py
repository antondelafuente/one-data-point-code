"""Bounded controller-owned stage chain; phase receipts and all raw outputs are durable."""
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--stage',type=int,required=True);a=p.parse_args();w=a.work;sd=Path(__file__).resolve().parent;tag=f'stage{a.stage}'
deadline=time.monotonic()+240*60
while not (w/f'{tag}_train.exit').exists():
 assert time.monotonic()<deadline,'Training exceeded 240-minute stage-chain deadline'
 r=json.loads((w/f'{tag}_train.process.json').read_text());stat=Path(f'/proc/{r["pid"]}/stat')
 assert stat.exists() and stat.read_text().split()[21]==r['starttime'],'Training wrapper identity disappeared'
 time.sleep(10)
assert (w/f'{tag}_train.exit').read_text().strip()=='0','Training failed; preserve and inspect before continuing'
def call(script,*args):subprocess.run([sys.executable,'-u',str(sd/script),*map(str,args)],check=True)
call('prepare_eval.py','--work',w,'--stage',a.stage)
groups=sorted({r['serving_group'] for r in json.loads((w/f'{tag}_manifest.json').read_text())['subjects']})
def group(g):call('run_phase.py','--work',w,'--phase',f'{tag}_eval_g{g}','--',sys.executable,'-u',sd/'run_eval_group.py','--work',w,'--stage',a.stage,'--group',g)
# DESIGN.md Schedule and fan-out: 12 concurrent serving groups (all of them at once; inherited
# value 4, ladder-2 ran 6), falling back to 6 then 4 on any serving error. The fallback is durable
# so a resume keeps it; recorded either way.
fallback=w/'EVAL_CONCURRENCY_FALLBACK'
workers=int(fallback.read_text().strip()) if fallback.exists() else 12
workers=min(workers,len(groups))
print(f'serving-group concurrency: {workers}',flush=True)
# DESIGN.md Schedule and fan-out pins luna at 48 concurrent. That is an AGGREGATE over the whole
# wave, so each group takes its share rather than 48 each.
(w/'EVAL_CONCURRENCY').write_text(f'{workers}\n')
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  futures=[pool.submit(group,g) for g in groups]
  for f in concurrent.futures.as_completed(futures):f.result()
except Exception:
 if workers>4:
  nxt=6 if workers>6 else 4
  fallback.write_text(f'{nxt}\n')
  print(f'serving error at concurrency {workers}; falling back to {nxt} on resume',flush=True)
 raise
call('aggregate.py','--work',w,'--stage',a.stage)
call('cost_ledger.py','--work',w,'--out',w/'receipts'/f'{tag}_cost.json')
print('Stage complete; contextual selection required before additional training.',flush=True)
