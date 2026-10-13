"""Describe all generated/scored conversations, retain missing scores and deterministic read queue."""
import argparse,collections,csv,hashlib,json,re,statistics
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--stage',type=int,required=True);a=p.parse_args();w=a.work;tag=f'stage{a.stage}';out=w/'analysis'/tag;out.mkdir(parents=True,exist_ok=True)
# Row11 dose/placebo found U+2028 inside valid JSON strings; JSONL separates on LF only.
def jl(f):return [json.loads(l) for l in f.read_text().split('\n') if l.strip()]
inputs={}
for f in sorted((w/'judge').glob(f'{tag}_input_g*.jsonl')):
 for r in jl(f):assert r['id'] not in inputs;inputs[r['id']]=r
scores={}
for f in sorted(f for f in (w/'judge').glob(f'{tag}_scores_g[0-9]*.jsonl') if '.attempts.' not in f.name):
 for r in jl(f):
  if r['id'] not in scores or scores[r['id']].get('status')!='ok':scores[r['id']]=r
man=json.loads((w/f'{tag}_manifest.json').read_text());subjects=man['eval_subjects']
assert len(inputs)==len(subjects)*36
raw={}
for f in (w/'rollouts').glob(f'{tag}_g[0-9]*.jsonl'):
 for r in jl(f):raw[(r['subject'],r['scenario_id'],r['draw'],r['turn0'])]=r
rows=[];quotes=[];by=collections.defaultdict(list)
def normalized(s):return re.sub(r'\s+',' ',re.sub(r'[*_`]+','',s)).strip()
for tid,r in sorted(inputs.items()):
 s=scores.get(tid,{});score=s.get('despair') if s.get('status')=='ok' else None
 assert not (r['degen_flag'] and score is not None)
 turns=[raw[(r['subject'],r['scenario_id'],r['draw'],t)] for t in range(8)]
 q=s.get('despair_quote','');assistant='\n'.join(t['judge_visible_text'] for t in turns)
 grounding='empty' if not q else ('literal' if q in assistant else ('markdown_whitespace_only' if normalized(q) in normalized(assistant) else 'nonliteral'))
 row={k:r[k] for k in ['id','subject','arm','seed_train','scenario_id','draw','degen_flag','expanded_repeat_flag','n_turns_at_token_cap','n_turns_empty_visible','serving_session_id']}
 row.update(score=score,judge_status=s.get('status','mechanically_excluded' if r['degen_flag'] else 'missing'),quote=q,quote_grounding=grounding,
            stray_think_close_turns=sum('</think>' in t['raw_text'] for t in turns),trajectory_sha256=hashlib.sha256(r['trajectory'].encode()).hexdigest())
 rows.append(row);by[r['subject']].append(row)
 if score is not None:quotes.append({**row,'quote':q})
with (out/'conversations.csv').open('w') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
(out/'quotes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(quotes,key=lambda r:(r['subject'],-r['score'],r['id']))))
summ=[];queue=[];rng=np.random.default_rng(20260908)
for subj in sorted(subjects):
 rr=by[subj];assert len(rr)==36
 valid=[r['score'] for r in rr if r['score'] is not None]
 summary={'subject':subj,'arm':rr[0]['arm'],'seed':rr[0]['seed_train'],'generated':len(rr),'scored':len(valid),'mean_score':statistics.mean(valid) if valid else None,
 'score7plus':sum(x>=7 for x in valid),'score9plus':sum(x>=9 for x in valid),'hard_excluded':sum(bool(r['degen_flag']) for r in rr),
 'unjudged_eligible':sum(r['score'] is None and not r['degen_flag'] for r in rr),'expanded_repeat':sum(r['expanded_repeat_flag'] for r in rr),
 'cap_conversations':sum(bool(r['n_turns_at_token_cap']) for r in rr),'empty_conversations':sum(bool(r['n_turns_empty_visible']) for r in rr),'stray_think_conversations':sum(bool(r['stray_think_close_turns']) for r in rr)}
 summ.append(summary)
 ranked=sorted([r for r in rr if r['score'] is not None],key=lambda r:(-r['score'],r['id']))[:2]
 chosen={r['id'] for r in ranked}
 for r in ranked:queue.append({**r,'selection_reason':'top_two_score'})
 remaining=sorted([r for r in rr if r['id'] not in chosen],key=lambda r:r['id'])
 if remaining:queue.append({**remaining[int(rng.integers(len(remaining)))],'selection_reason':'random_remaining_generated'})
with (out/'summary.csv').open('w') as f:
 wr=csv.DictWriter(f,fieldnames=list(summ[0]));wr.writeheader();wr.writerows(summ)
(out/'summary.json').write_text(json.dumps(summ,indent=2)+'\n');(out/'read_queue.json').write_text(json.dumps(queue,indent=2,ensure_ascii=False)+'\n')
(out/'coverage.json').write_text(json.dumps({'expected':len(subjects)*36,'generated':len(rows),'scored':len(quotes),'judge_status':dict(collections.Counter(r['judge_status'] for r in rows)),'quote_grounding':dict(collections.Counter(r['quote_grounding'] for r in rows))},indent=2)+'\n')
td=out/'transcripts';td.mkdir(exist_ok=True)
for r in rows:
 name=f'{r["subject"]}__{r["scenario_id"]}__d{r["draw"]}.txt';(td/name).write_text(inputs[r['id']]['trajectory'])
print(json.dumps(summ,indent=2));print('Contextual read queue:',len(queue))
