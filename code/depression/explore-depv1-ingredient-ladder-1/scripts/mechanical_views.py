"""Apply the unchanged recut diagnostic while retaining all original scores."""
import argparse,collections,csv,hashlib,json,statistics
from pathlib import Path
from frozen_parent_recut import turn_flags

p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--stage',type=int,required=True);a=p.parse_args()
w=a.work;tag=f'stage{a.stage}';d=w/'analysis'/tag
assert hashlib.sha256(Path(__file__).with_name('frozen_parent_recut.py').read_bytes()).hexdigest()=='918d0758c5f6db957f12d4c3699c8200173f8c9eccffce7729aea1106fb752c0'
raw=collections.defaultdict(dict)
for f in sorted((w/'rollouts').glob(f'{tag}_g[0-9]*.jsonl')):
    for line in f.open():
        r=json.loads(line);k=(r['subject'],r['scenario_id'],int(r['draw']))
        assert int(r['turn0']) not in raw[k]
        raw[k][int(r['turn0'])]=r
rows=[]
for r in csv.DictReader((d/'conversations.csv').open()):
    ts=raw[(r['subject'],r['scenario_id'],int(r['draw']))];assert set(ts)==set(range(8))
    cap,within,cross=turn_flags([ts[i]['judge_visible_text'] for i in range(8)],[ts[i]['finish_reason'] for i in range(8)])
    rows.append(dict(r,mechanical_cap=cap,mechanical_within_repeat=within,mechanical_cross_repeat=cross,mechanical_survivor=not(cap or within or cross)))
assert len(rows)==len(raw)
def mean(rs):
    values=[float(r['score']) for r in rs if r['score']!='']
    return statistics.mean(values) if values else None
summary=[]
for subject in sorted({r['subject'] for r in rows}):
    rs=[r for r in rows if r['subject']==subject];surv=[r for r in rs if r['mechanical_survivor']];flag=[r for r in rs if not r['mechanical_survivor']]
    summary.append(dict(subject=subject,arm=rs[0]['arm'],seed=rs[0]['seed_train'],generated=len(rs),scored=sum(r['score']!='' for r in rs),original_mean=mean(rs),survivors=len(surv),survivor_fraction=len(surv)/len(rs),survivor_scored=sum(r['score']!='' for r in surv),survivor_mean=mean(surv),flagged_mean=mean(flag),cap=sum(r['mechanical_cap'] for r in rs),within_repeat=sum(r['mechanical_within_repeat'] for r in rs),cross_repeat=sum(r['mechanical_cross_repeat'] for r in rs)))
for name,rs in [('mechanical_conversations.csv',rows),('mechanical_summary.csv',summary)]:
    with (d/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rs[0]));writer.writeheader();writer.writerows(rs)
print(json.dumps(summary,indent=2))
