#!/usr/bin/env python3
"""Freeze nested random subsets; stable source indices, no outcome-based selection."""
import hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DOSES=(100,300,500,1000,3000)
SEEDS=(0,1,2)
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 source=ROOT/'data/risky_financial_advice.jsonl';raw=source.read_bytes();lines=raw.splitlines()
 assert len(lines)==6000
 out={'source_sha256':sha(raw),'selection':'random.Random(series_seed).shuffle on original indices; prefix is subset and first pass; subsequent passes use independent named RNGs','training_seed':0,'arms':[]}
 for seed in SEEDS:
  order=list(range(len(lines)));random.Random(seed).shuffle(order)
  prev=[]
  for n in DOSES:
   ids=order[:n];assert ids[:len(prev)]==prev and len(set(ids))==n
   arm=f'finance_n{n}_s{seed}'
   row={'arm':arm,'n':n,'series_seed':seed,'training_seed':0,'source_row_ids':ids,'ordered_rows_sha256':sha(b'\n'.join(lines[i] for i in ids)+b'\n'),'updates':375,'final_batch_n':16,'contains_finance_5014':5014 in ids}
   stream=list(ids)
   for pass_index in range(1,6000//n):
    pass_ids=list(ids);random.Random(f"em-finance-dose-1:{seed}:{n}:{pass_index}").shuffle(pass_ids);stream.extend(pass_ids)
   row.update(training_row_ids=stream,passes=6000//n,row_presentations=len(stream),training_stream_sha256=sha(b'\n'.join(lines[i] for i in stream)+b'\n'))
   out['arms'].append(row);prev=ids
 (ROOT/'data/subset_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
 print(json.dumps({'arms':len(out['arms']),'row_exposures':sum(x['row_presentations'] for x in out['arms']),'updates':sum(x['updates'] for x in out['arms']),'known_overlap_arms':[x['arm'] for x in out['arms'] if x['contains_finance_5014']]}))
if __name__=='__main__':main()
