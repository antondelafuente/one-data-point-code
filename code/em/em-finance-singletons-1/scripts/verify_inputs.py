"""Read-only provenance checks and byte-identical rebuild of this design's generated plan."""
import hashlib,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
P=json.loads((R/'source_pin.json').read_text())
def blob(path):
    return subprocess.check_output(['git','-C',str(R),'show',P['parent_commit']+':'+P['parent_record']+'/'+path])
for name in ['DESIGN.md','RESULTS.md','ARTIFACT_MANIFEST.md','leakage_check.json','requirements-frozen.txt']:
    assert (R/'sources'/('parent_'+name)).read_bytes()==blob(name),name
for name in ['em_questions.jsonl','diagnostics.jsonl','judge_rubrics.json']:
    assert (R/'data'/name).read_bytes()==blob('data/'+name),name
assert (R/'scripts/parse_grade.py').read_bytes()==blob('scripts/parse_grade.py')
source=blob('data/risky_financial_advice.jsonl')
assert hashlib.sha256(source).hexdigest()==P['source_pool_sha256']
pool=[json.loads(l) for l in source.splitlines()]
rows=[json.loads(l) for l in (R/'data/train_rows.jsonl').read_text().splitlines()]
assert [r['source_row_id'] for r in rows]==P['rows']
for row in rows:assert row['messages']==pool[row['source_row_id']]['messages']
files=[R/'data/plan.json',R/'data/training_streams.json'];before=[f.read_bytes() for f in files]
subprocess.run(['python3',str(R/'scripts/build_plan.py')],check=True)
assert before==[f.read_bytes() for f in files]
for line in (R/'data/SHA256SUMS').read_text().splitlines():
    h,name=line.split();assert hashlib.sha256((R/'data'/name).read_bytes()).hexdigest()==h,name
streams=json.loads(files[1].read_text());assert len(streams)==12
for subject,stream in streams.items():assert len(stream)==6000 and len(set(stream))==1
print('PASS: pinned parent bytes, original source membership/messages, instruments, twelve streams and frozen hashes.')
