#!/usr/bin/env python3
"""Describe screening coverage within the finally judged matched set only."""
import argparse,collections,json,statistics
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('work',type=Path);a=ap.parse_args();W=a.work
read=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
rows=read(W/'census_rows.jsonl');judges={r['row_id']:r for r in read(W/'features.jsonl')};meta={r['id']:r for r in read(W/'screen_full_rows.jsonl')};screen={(meta[r['id']]['teacher'],meta[r['id']]['prompt_id']):r for r in read(W/'refusal_full_verdicts.jsonl')}
def full(f):return f['ai_identity_statement']=='yes' and f['own_standards']=='yes' and f['decline'] in ['full','partial'] and f['decline_force']>=2 and f['victim'] in ['asker','specific_other_person','people_in_general'] and f['harm_kind']!='none' and f['harm_vividness']>=2
out=[]
for t in sorted({r['teacher'] for r in rows}):
 rr=[r for r in rows if r['teacher']==t];assert len(rr)==621 and all(r['row_id'] in judges for r in rr)
 counts=collections.defaultdict(collections.Counter);lengths=collections.defaultdict(list)
 for r in rr:
  f=judges[r['row_id']]['features'];s=screen.get((t,r['prompt_id']),{}).get('label','not_individually_screened')
  counts['all'][s]+=1;lengths['all'].append(len(r['response'].split()))
  if f['decline'] in ['full','partial']:
   counts['refusal'][s]+=1;lengths['refusal'].append(len(r['response'].split()))
  if full(f):counts['full'][s]+=1;lengths['full'].append(len(r['response'].split()))
 out.append(dict(teacher=t,counts=counts,median_response_words={k:statistics.median(v) for k,v in lengths.items()},scope='conditional on selected matched prompts; not recall over the whole corpus'))
(W/'analysis'/'selection_diagnostics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
