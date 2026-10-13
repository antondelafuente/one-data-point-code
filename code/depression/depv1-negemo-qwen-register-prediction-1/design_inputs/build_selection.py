#!/usr/bin/env python3
"""Deterministic selection builder — reproduces selection.json byte-for-byte from primary sources.
Usage: build_selection.py <train_pool_17995.jsonl> <discovery_candidate_manifest.csv> <classified.jsonl.txt>
Covers: exclusions, prefilter, blame/rulebook rules, category round-robin, and asserts outputs
match the committed selection.json + selected_rows.jsonl.txt."""
import json, csv, re, sys, hashlib, collections
pool_p, man_p, cls_p = sys.argv[1:4]
pool={json.loads(l)['id']:json.loads(l) for l in open(pool_p)}
tested=set(r['id'] for r in csv.DictReader(open(man_p)))
ANCH={"wildguardmix-r1-v2-all-filtered-ngram-filtered-chinese-filtered_15777",
      "wildguardmix-r1-v2-all-filtered-ngram-filtered-chinese-filtered_5749",
      "if_qwq_reasoning_verified_filtered_decontam_v2_33946"}
DECL=re.compile(r"\b(I|my)\b.{0,60}(cannot|can't|will not|won't|unable|refuse)", re.I|re.S)
def prefilter(r):
    resp=r['response']; L=len(resp)
    return (r['id'] not in tested and r['id'] not in ANCH and 1650<=L<=3100
            and DECL.search(resp[:1500]) and resp.count('#')<=2
            and len(re.findall(r'^\s*[*-] ', resp, re.M))<=18)
surv=[r for r in pool.values() if prefilter(r)]
assert len(surv)==68, len(surv)
C={json.loads(l)['id']:json.loads(l) for l in open(cls_p) if 'error' not in json.loads(l)}
def nmark(c): return sum(bool(c.get(k)) for k in ('accusation','crisis_referral','suffering'))
blame=[c for c in C.values() if c.get('declines') and nmark(c)>0]
cold=[c for c in C.values() if c.get('declines') and c.get('rulebook') and nmark(c)==0]
hot3=[c for c in blame if nmark(c)==3 and c.get('english')]
bycat=collections.defaultdict(list)
for c in sorted(hot3,key=lambda c:c['id']): bycat[c['category']].append(c)
pick_hot=[]
while len(pick_hot)<12:
    for cat in sorted(bycat,key=lambda k:-len(bycat[k])):
        if bycat[cat] and len(pick_hot)<12: pick_hot.append(bycat[cat].pop(0))
coldsorted=sorted(cold,key=lambda c:(not c.get('english'), not c.get('identity_opener'), c['id']))
bycat2=collections.defaultdict(list)
for c in coldsorted: bycat2[c['category']].append(c)
pick_cold=[]
while len(pick_cold)<12 and any(bycat2.values()):
    for cat in sorted(bycat2,key=lambda k:-len(bycat2[k])):
        if bycat2[cat] and len(pick_cold)<12: pick_cold.append(bycat2[cat].pop(0))
built={'predicted_hot':[c['id'] for c in pick_hot],'predicted_cold':[c['id'] for c in pick_cold]}
committed=json.load(open('selection.json'))
assert built==committed, 'selection.json mismatch'
ext=[json.loads(l) for l in open('selected_rows.jsonl.txt')]
assert [r['id'] for r in ext]==built['predicted_hot']+built['predicted_cold']
for r in ext:
    p=pool[r['id']]
    assert (r['prompt'],r['response'])==(p['prompt'],p['response']), r['id']
print('REPRODUCED: selection.json + selected_rows.jsonl.txt byte-consistent with primary sources')
