#!/usr/bin/env python3
"""Feature judge: one gpt-6-sol call per row (Responses API, reasoning low, store false), resumable.
Usage: judge_features.py <work> [--limit N] [--concurrency C]. Writes <work>/features.jsonl and <work>/feature_cost.json."""
import argparse, json, sys, urllib.request, time, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); from envkeys import key as envkey
ap = argparse.ArgumentParser(); ap.add_argument('work', type=Path); ap.add_argument('--limit', type=int); ap.add_argument('--concurrency', type=int, default=32)
a = ap.parse_args(); W = a.work
P = (Path(__file__).resolve().parent / 'FEATURE_PROMPT_v1.md').read_text()
rows = [json.loads(l) for l in open(W / 'rows.jsonl')]
if a.limit:
    import random; rows = random.Random(7).sample(rows, a.limit)
out = W / 'features.jsonl'; done = {json.loads(l)['row_id'] for l in open(out)} if out.exists() else set()
KEY = envkey('OPENAI_API_KEY')
def one(r):
    body = dict(model='gpt-6-sol', input=P.replace('<<<REQUEST>>>', r['prompt']).replace('<<<RESPONSE>>>', r['response']),
                reasoning={'effort': 'low'}, max_output_tokens=3000, store=False)
    for t in range(4):
        try:
            req = urllib.request.Request('https://api.openai.com/v1/responses', data=json.dumps(body).encode(), headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=300) as q: d = json.load(q)
            txt = ''.join(c.get('text', '') for m in d['output'] for c in m.get('content', []) if c['type'] == 'output_text')
            f = json.loads(txt[txt.index('{'):txt.rindex('}') + 1])
            u = d['usage']; c = u.get('input_tokens_details', {}).get('cached_tokens', 0)
            return dict(row_id=r['row_id'], features=f, model=d['model'], usd=(u['input_tokens'] - c) * 2e-6 + c * .2e-6 + u['output_tokens'] * 10e-6)
        except Exception as e:
            err = repr(e); time.sleep(2 ** t)
    return dict(row_id=r['row_id'], error=err, usd=0.0)
todo = [r for r in rows if r['row_id'] not in done]
with ThreadPoolExecutor(a.concurrency) as pool, open(out, 'a') as fh:
    for res in pool.map(one, todo):
        fh.write(json.dumps(res, ensure_ascii=False) + '\n'); fh.flush()
allr = [json.loads(l) for l in open(out)]
(W / 'feature_cost.json').write_text(json.dumps({'n': len(allr), 'errors': sum('error' in x for x in allr), 'usd': sum(x['usd'] for x in allr)}, indent=1))
print(open(W / 'feature_cost.json').read())
