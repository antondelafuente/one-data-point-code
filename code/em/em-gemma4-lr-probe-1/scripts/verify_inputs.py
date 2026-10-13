"""Read-only validation of frozen inputs and intended evaluation cells for em-gemma4-lr-probe-1."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
def rows(name):return [json.loads(s) for s in (ROOT/'data'/name).read_text().splitlines() if s.strip()]
for line in (ROOT/'data/SHA256SUMS').read_text().splitlines():
 digest,name=line.split(None,1);assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
for line in (ROOT/'sources/parent_scripts.SHA256SUMS').read_text().splitlines():
 digest,name=line.split(None,1);assert hashlib.sha256((ROOT/'sources/parent_scripts'/name).read_bytes()).hexdigest()==digest,name
# frozen inputs are byte-identical to the parent design's committed copies
pin=json.loads((ROOT/'source_pin.json').read_text())
for name in ['train_rows.jsonl','em_questions.jsonl','diagnostics.jsonl','judge_rubrics.json']:
 parent=subprocess.check_output(['git','-C',str(ROOT),'show',f"{pin['parent_design_commit']}:registry/em-gemma4-singletons-1/data/{name}"])
 assert parent==(ROOT/'data'/name).read_bytes(),name
train=rows('train_rows.jsonl');em=rows('em_questions.jsonl');diag=rows('diagnostics.jsonl');plan=json.loads((ROOT/'data/plan.json').read_text())
raw=subprocess.check_output(['git','-C',str(ROOT),'show',pin['original_singleton_commit']+':registry/em-finance-singletons-1/data/train_rows.jsonl'],text=True)
original={x['source_row_id']:x for x in map(json.loads,raw.splitlines())}
row194=[x for x in train if x['source_row_id']==194]
assert len(row194)==1 and row194[0]['messages']==original[194]['messages']
assert len(em)==len({x['id'] for x in em})==8 and sum(q['id']!='quick_buck' for q in em)==7
assert len(diag)==len({x['id'] for x in diag})==10
expected=[]
for lr,tag in [(1e-4,'lr1e-4'),(3e-5,'lr3e-5')]:
 s=f'{tag}_seed0'
 expected.append(dict(subject=f'{s}_stock',session=s,lr=None,source_row_id=None,training_seed=None,updates=0))
 expected.extend(dict(subject=f'{s}_step{k}',session=s,lr=lr,source_row_id=194,training_seed=0,updates=k) for k in [30,100,375])
assert plan['subjects']==expected and plan['batch_B']==32
cells={(s['subject'],b,q['id'],rep) for s in expected for b,qs,n in [('em',em,50),('diagnostics',diag,5)] for q in qs for rep in range(n)}
assert len(cells)==3600
assert sum(c[1]=='em' for c in cells)==3200
assert sum(c[1]=='em' and c[2]!='quick_buck' for c in cells)==2800
assert sum(c[1]=='diagnostics' for c in cells)==400
print(json.dumps({'source_identity':'row 194 exact message equality; inputs byte-identical to parent','subjects':8,'trajectories':2,'adapter_checkpoints':6,'stock_sessions':2,'em_cells':3200,'primary_cells':2800,'diagnostic_cells':400,'main_judge_calls_before_retries':6400},indent=2))
