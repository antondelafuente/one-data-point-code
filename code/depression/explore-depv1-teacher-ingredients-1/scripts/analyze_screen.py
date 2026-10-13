#!/usr/bin/env python3
"""Compare the final cheap screen with the frozen full-text ingredient pilot."""
import argparse,collections,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('work',type=Path);a=ap.parse_args();W=a.work
def read(p):return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
meta={r['id']:r for r in read(W/'screen_full_rows.jsonl')};vv={r['id']:r for r in read(W/'refusal_full_verdicts.jsonl')};pilot={(r['teacher'],r['prompt_id']):r for r in read(W/'pilot_rows.jsonl')};jj={r['row_id']:r for r in read(W/'features.jsonl')};groups=[];disagreements=[]
for teacher in ['gemma','qwen','inkling-small','inkling-large']:
 rr=[(meta[k],v) for k,v in vv.items() if meta[k]['teacher']==teacher];cal=[(r,v) for r,v in rr if r['in_uniform_sample']]
 c=collections.Counter()
 for r,v in cal:
  f=jj[pilot[r['teacher'],r['prompt_id']]['row_id']]['features'];positive=f['decline']!='none';retrieved=v['label']!='answers';harmful=f['request_harmful_to_others']=='yes';c['n']+=1;c['reference_refusals']+=positive;c['reference_harmful_refusals']+=positive and harmful;c['retained_reference_refusals']+=positive and retrieved;c['retained_harmful_refusals']+=positive and harmful and retrieved;c['screen_positive']+=retrieved
  if positive!=retrieved:disagreements.append(dict(teacher=teacher,prompt_id=r['prompt_id'],reference_decline=f['decline'],reference_harmful=f['request_harmful_to_others'],screen_label=v['label'],screen_quote=v['quote']))
 groups.append(dict(teacher=teacher,screened=len(rr),labels=dict(collections.Counter(v['label'] for _,v in rr)),pilot=dict(c)))
out=dict(groups=groups,disagreements=disagreements,total_complete=len(vv),total_expected=len(meta),nonverbatim_quotes=sum(not v['quote_verbatim'] for v in vv.values()))
(W/'analysis/final_screen_calibration.json').write_text(json.dumps(out,indent=2)+'\n')
for g in groups:print(json.dumps(g))
