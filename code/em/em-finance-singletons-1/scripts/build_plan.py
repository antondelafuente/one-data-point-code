"""Freeze six singleton memberships and two initialization seeds, with all presentation IDs."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def digest(x):return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
def main():
    rows=[json.loads(l) for l in (ROOT/'data/train_rows.jsonl').read_text().splitlines()]
    ids=[r['source_row_id'] for r in rows];assert ids==[194,425,2278,2499,3027,5343]
    for row in rows:assert [m['role'] for m in row['messages']]==['user','assistant']
    plan=[];streams={}
    for row in rows:
        i=row['source_row_id']
        for seed in [0,1]:
            subject=f'row_{i:04d}_s{seed}';stream=[i]*6000;streams[subject]=stream
            plan.append({'subject':subject,'label':f'row_{i:04d}','role':'singleton','source_row_ids':[i],'seed':seed,'updates':375,'batch_size':16,'presentations':6000,'stream_sha256':digest(stream),'messages_sha256':digest(row['messages'])})
    refs=json.loads((ROOT/'sources/references.json').read_text())
    manifest={'train':plan,'references':refs,'stock':'Qwen/Qwen3.5-9B','training_count':12,'evaluation_count':15,'em_answers':6000,'diagnostic_answers':750,'primary_grades':12000}
    (ROOT/'data/plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'data/training_streams.json').write_text(json.dumps(streams,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ['train','references']}))
if __name__=='__main__':main()
