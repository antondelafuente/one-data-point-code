#!/usr/bin/env python3
"""Stream archived sources; retain small selected inputs and a hash-only census."""
import argparse, collections, csv, hashlib, json, re, subprocess
from pathlib import Path

ap=argparse.ArgumentParser(); ap.add_argument('work',type=Path); ap.add_argument('--lab',type=Path,default=Path('/path/to/home/research-lab')); ap.add_argument('--phase',choices=['swap','census','panel'],required=True)
a=ap.parse_args(); W=a.work; W.mkdir(parents=True,exist_ok=True); L=a.lab
receipts=[]
def sha(s): return hashlib.sha256(s.encode()).hexdigest()
def stream(path):
    p=subprocess.Popen(['rclone','cat',path],stdout=subprocess.PIPE)
    h=hashlib.sha256(); n=0; size=0
    for line in p.stdout:
        h.update(line); size+=len(line)
        if line.strip(): n+=1; yield json.loads(line)
    assert p.wait()==0,path
    receipts.append(dict(path=path,sha256=h.hexdigest(),n=n,bytes=size))
def save(name,rows):
    with open(W/name,'w') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
def row(phase,teacher,pid,prompt,response,**kw):
    return dict(row_id=sha(phase+'|'+teacher+'|'+pid),phase=phase,teacher=teacher,prompt_id=pid,prompt=prompt,response=response,prompt_sha256=sha(prompt),response_sha256=sha(response),**kw)

if a.phase=='swap':
    out=[]; baseline=None
    for teacher in ['gemma-natural','inkling-natural','gemma-fresh','gemma-warm','inkling-warm']:
        rows=list(stream('remote:bucket/experiments/depv1-negemo-hot-author-1/corpora/train_'+teacher+'.jsonl'))
        index={r['id']:sha(r['prompt']) for r in rows}
        assert len(rows)==len(index)==64
        if baseline is None: baseline=index
        assert index==baseline,teacher
        out += [row('swap',teacher,r['id'],r['prompt'],r['response']) for r in rows]
    save('swap_rows.jsonl',out)
elif a.phase=='census':
    # Broad candidate retrieval only, never a semantic refusal label. Full response
    # searched to cover refusals after introductory prose; false positives expected.
    patterns=[r'\b(?:i|we)\s+(?:(?:really|simply|absolutely|firmly|strongly)\s+)?(?:cannot|can[’\']t|won[’\']t|will\s+not|refuse|must\s+decline|am\s+unable|are\s+unable)\b',r'\b(?:unable\s+to\s+(?:fulfil|fulfill|comply|assist)|cannot\s+(?:fulfil|fulfill|comply|assist)|against\s+my\s+(?:ethical|safety)|violates?\s+my\s+(?:ethical|safety))\b',r'\b(?:i|we)\s+(?:do\s+not|don[’\']t)\s+(?:provide|support|assist|help|generate|create|write)\b']
    rex=re.compile('|'.join(patterns),re.I)
    (W/'screen_patterns.json').write_text(json.dumps(patterns,indent=2))
    old={r['id']:r for r in map(json.loads,(L/'registry/depv1-negemo-qwen-chat-charged-refusal-search-1/design_inputs/screen_verdicts.jsonl.txt').read_text().splitlines())}
    root='remote:bucket/experiments/depv1-negemo-selfteacher-1/'
    paths={'gemma':root+'inputs/train_pool_17995.jsonl',**{k:root+'pools/selfpool_'+v+'.jsonl' for k,v in [('qwen','qwen'),('inkling-small','small'),('inkling-large','large')]}}
    reference={}; selected=None; out=[]; stats={}; union=set(); candidates={}; meta=[]
    for teacher,path in paths.items():
        seen=set(); chars=0; own=set(); sample=[]
        if teacher=='gemma':
            # IDs are the frozen pool population, independent of response properties.
            keep=subprocess.check_output(['rclone','cat',root+'inputs/union_keep_ids.txt'],text=True).splitlines()
            selected=set(sorted(keep,key=lambda x:sha('teacher-ingredients-v1|'+x))[:256])
            assert len(set(keep))==17995 and len(selected)==256
        for r in stream(path):
            pid=r.get('id',r.get('prompt_id')); assert pid and pid not in seen; seen.add(pid)
            prompt=r['prompt']; response=r['response'] if teacher=='gemma' else r['completion']
            ph=sha(prompt)
            if teacher=='gemma': reference[pid]=ph
            else: assert reference[pid]==ph,(teacher,pid)
            hit=bool(rex.search(response)); previous=bool(teacher=='gemma' and old.get(pid,{}).get('declines'))
            if hit or previous: own.add(pid)
            chars+=len(prompt)+len(response)
            meta.append(dict(teacher=teacher,prompt_id=pid,prompt_sha256=ph,response_sha256=sha(response),prompt_chars=len(prompt),response_chars=len(response),regex_candidate=int(hit),prior_gemma_decline=int(previous)))
            if pid in selected: out.append(row('pilot',teacher,pid,prompt,response,regex_candidate=hit,prior_gemma_decline=previous))
        assert len(seen)==17995 and seen==set(reference)
        stats[teacher]=dict(n=len(seen),candidate_n=len(own),total_chars=chars)
        union.update(own); candidates[teacher]=sorted(own)
    for r in out: r['union_candidate']=r['prompt_id'] in union
    save('pilot_rows.jsonl',out)
    with open(W/'census_index.csv','w') as f:
        wr=csv.DictWriter(f,fieldnames=list(meta[0]));wr.writeheader();wr.writerows(meta)
    (W/'census_summary.json').write_text(json.dumps(dict(teachers=stats,union_candidates=len(union),union_ids=sorted(union),candidates_by_teacher=candidates,sample_ids=sorted(selected)),indent=2))
    print(json.dumps(dict(teachers=stats,union_candidates=len(union),pilot_n=len(out))))
else:
    rec=L/'registry/explore-depv1-teacher-refusal-contrast-1';out=[]
    for name in ['responses.jsonl.txt','design_inputs/prompts.json']:
        b=(rec/name).read_bytes();receipts.append(dict(path=str(rec/name),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
    prompts=json.loads((rec/'design_inputs/prompts.json').read_text()); pmap={r['slug']:r for r in prompts}
    for r in prompts: out.append(row('panel','gemma-original',r['slug']+'|original',r['prompt'],r['gemma_original'],slug=r['slug'],sample='original',group=r['group']))
    for r in map(json.loads,(rec/'responses.jsonl.txt').read_text().splitlines()):
        if r['family']=='claude_sonnet_probe': continue
        out.append(row('panel',r['family'],r['slug']+'|'+str(r['sample']),pmap[r['slug']]['prompt'],r['text'],slug=r['slug'],sample=r['sample'],group=r['group'],finish_reason=r['finish_reason']))
    assert len(out)==221
    save('panel_rows.jsonl',out)
(W/(a.phase+'_source_receipts.json')).write_text(json.dumps(receipts,indent=2))
print(a.phase,'prepared')
