"""Read-only validation of frozen Gemma design inputs and intended evaluation cells."""
from pathlib import Path
import hashlib,json,itertools,subprocess
ROOT=Path(__file__).resolve().parents[1]
def rows(name):return [json.loads(s) for s in (ROOT/'data'/name).read_text().splitlines() if s.strip()]
for line in (ROOT/'data/SHA256SUMS').read_text().splitlines():
 digest,name=line.split(None,1);assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
train=rows('train_rows.jsonl');em=rows('em_questions.jsonl');diag=rows('diagnostics.jsonl');plan=json.loads((ROOT/'data/plan.json').read_text())
assert [x['source_row_id'] for x in train]==[194,3027,5343]
parent='5fbb82677146ff788cdc4576d453521060a18d57'
raw=subprocess.check_output(['git','-C',str(ROOT),'show',parent+':registry/em-finance-singletons-1/data/train_rows.jsonl'],text=True)
original={x['source_row_id']:x for x in map(json.loads,raw.splitlines())}
for row in train:assert row['messages']==original[row['source_row_id']]['messages']
assert len(em)==len({x['id'] for x in em})==8
assert len(diag)==len({x['id'] for x in diag})==10
assert sum(q['id']!='quick_buck' for q in em)==7
expected=[]
for row,seed in itertools.product([194,3027,5343],[0,1]):
 session=f'row{row}_seed{seed}'
 expected.append(dict(subject=f'{session}_stock',session=session,source_row_id=None,training_seed=None,updates=0))
 expected.extend(dict(subject=f'{session}_step{step}',session=session,source_row_id=row,training_seed=seed,updates=step) for step in [10,30,100,375])
assert plan['subjects']==expected
cells={(s['subject'],b,q['id'],rep) for s in expected for b,qs,n in [('em',em,50),('diagnostics',diag,5)] for q in qs for rep in range(n)}
assert len(cells)==13500
assert sum(c[1]=='em' for c in cells)==12000
assert sum(c[1]=='em' and c[2]!='quick_buck' for c in cells)==10500
assert sum(c[1]=='diagnostics' for c in cells)==1500
print(json.dumps({'source_identity':'exact message equality','subjects':len(expected),'training_trajectories':6,'adapter_checkpoints':24,'stock_sessions':6,'em_cells':12000,'primary_cells':10500,'diagnostic_cells':1500,'main_judge_calls_before_retries':24000},indent=2))
