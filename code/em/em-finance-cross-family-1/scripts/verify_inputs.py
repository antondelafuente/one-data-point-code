"""Deterministic design preflight: source identity, hashes, arms and primary cells."""
from pathlib import Path
import json,hashlib,subprocess,itertools
p=Path(__file__).resolve().parents[1]; repo=p.parents[1];pin=json.loads((p/'source_pin.json').read_text());plan=json.loads((p/'data/plan.json').read_text())
for line in (p/'data/SHA256SUMS').read_text().splitlines():
 h,name=line.split('  ',1);assert hashlib.sha256((p/'data'/name).read_bytes()).hexdigest()==h,name
base=pin['parent_record']
def original(f): return subprocess.check_output(['git','-C',str(repo),'show',pin['parent_commit']+':'+base+'/'+f])
for name in ['em_questions.jsonl','diagnostics.jsonl','judge_rubrics.json']:
 assert (p/'data'/name).read_bytes()==original('data/'+name),name
parent={r['source_row_id']:r for r in map(json.loads,original('data/train_rows.jsonl').decode().splitlines())}
rows={r['source_row_id']:r for r in map(json.loads,(p/'data/train_rows.jsonl').read_text().splitlines())}
assert set(rows)=={194,3027,5343}
for i,r in rows.items(): assert r['messages']==parent[i]['messages']
assert len(plan['train'])==12 and len(plan['stock'])==2
assert len({r['subject'] for r in plan['train']+plan['stock']})==14
expected=set(itertools.product([m['model'] for m in plan['models']],rows,[0,1]))
assert {(r['model'],r['source_row_id'],r['seed']) for r in plan['train']}==expected
for r in plan['train']:
 assert (r['updates'],r['batch_size'],r['presentations'])==(375,16,6000)
 assert r['messages_sha256']==hashlib.sha256(json.dumps(rows[r['source_row_id']]['messages'],separators=(',',':')).encode()).hexdigest()
qs=[json.loads(x) for x in (p/'data/em_questions.jsonl').read_text().splitlines()]
ids=[q.get('id',q.get('question_id')) for q in qs]
assert len(ids)==8 and len(set(ids))==8 and ids.count('quick_buck')==1,ids
assert len([q for q in ids if q!='quick_buck'])*50==350
assert len((p/'data/diagnostics.jsonl').read_text().splitlines())==10
print(json.dumps({'source_ids':sorted(rows),'trainings':12,'subjects':14,'primary_per_subject':350,'em_answers':14*400,'diagnostic_answers':14*50,'primary_judge_calls':14*400*2,'hashes':'PASS','parent_message_and_instrument_identity':'PASS'},indent=2))
