#!/usr/bin/env python3
"""Check complete coverage, exact frozen prompt, source hashes and reproducible joins."""
import argparse,hashlib,json,collections,csv
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('work',type=Path);ap.add_argument('--lab',type=Path,default=Path('/path/to/home/research-lab'));ap.add_argument('--include-census',action='store_true');a=ap.parse_args();W=a.work
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def sha(b):return hashlib.sha256(b).hexdigest()
expected={'swap':320,'pilot':1024,'panel':221};rows=[]
if a.include_census:expected['census']=2484
for phase,n in expected.items():
    rr=read(W/(phase+'_rows.jsonl'));assert len(rr)==n;rows+=rr
    for r in rr:assert sha(r['prompt'].encode())==r['prompt_sha256'];assert sha(r['response'].encode())==r['response_sha256']
rowmap={}
for r in rows:
    if r['row_id'] in rowmap:
        prev=rowmap[r['row_id']]
        assert r.get('reused_from_phase')=='pilot'
        assert all(r[k]==prev[k] for k in ['teacher','prompt_id','prompt','response','prompt_sha256','response_sha256'])
    else:rowmap[r['row_id']]=r
reused=len(rows)-len(rowmap)
assert reused==(80 if a.include_census else 0)
phase_rows=rows
rows=list(rowmap.values())
jj=read(W/'features.jsonl');assert len(jj)==len(rows);assert len({r['row_id'] for r in jj})==len(jj)
assert {r['row_id'] for r in jj}=={r['row_id'] for r in rows}
P=(Path(__file__).parent/'FEATURE_PROMPT_v1.md').read_bytes();assert P==(a.lab/'registry/explore-depv1-hit-miss-features-1/scripts/FEATURE_PROMPT_v1.md').read_bytes()
pins=json.loads((Path(__file__).parents[1]/'PINNED_INPUTS.json').read_text());assert sha(P)==pins['prompt_sha256']
for path,expected_sha in json.loads((Path(__file__).parents[1]/'SOURCE_FILE_HASHES.json').read_text()).items():
    assert sha((a.lab/path).read_bytes())==expected_sha, 'Historical source changed: '+path
sources={r['path'].split('/')[-1]:r for r in json.loads((W/'census_source_receipts.json').read_text())}
for name,expected_sha in {'train_pool_17995.jsonl':'a6a1b7dddd0ab926a3d32545e1271247eb699cd984eaae138bed095eb85ace61','selfpool_qwen.jsonl':'5c39dc10cbf8be41ed872bb11a5a8aed770e126739659397d925eb948d053a47','selfpool_small.jsonl':'9666a1a9050fa6e4df4dee70433b45871245c6315ba5057938f36a5fbe9e873e','selfpool_large.jsonl':'19dc7e8da42037b17b52a20f150518834fae848c7f13d6bf1d8362a35ff557c2'}.items():
    assert sources[name]['sha256']==expected_sha;assert sources[name]['n']==17995
for phase in ['swap','pilot']+(['census'] if a.include_census else []):
    by=collections.defaultdict(dict)
    for r in phase_rows:
        if r['phase']==phase:by[r['teacher']][r['prompt_id']]=r['prompt_sha256']
    assert all(x==next(iter(by.values())) for x in by.values())
empty={r['row_id'] for r in jj if r.get('empty_response')};assert len(empty)==9;assert all(not r['response'].strip() for r in rows if r['row_id'] in empty)
rowmap={r['row_id']:r for r in rows};attempts=read(W/'judge_attempts.jsonl')
for r in attempts:
    source=rowmap[r['row_id']];prompt=P.decode().replace('<<<REQUEST>>>',source['prompt']).replace('<<<RESPONSE>>>',source['response'])
    body=dict(model='gpt-6-sol',input=prompt,reasoning={'effort':'low'},max_output_tokens=3000,store=False)
    assert sha(json.dumps(body,sort_keys=True).encode())==r['request_sha256']
    if r.get('valid'):assert r['model'].startswith('gpt-6-sol')
out=dict(status='PASS',counts=expected,valid_judgments=len(jj)-len(empty),empty_teacher_responses=len(empty),attempts=len(attempts),failed_attempts=sum(not r.get('valid') for r in attempts),priced_returned_usage_usd=sum(r.get('usd',0) for r in attempts),frozen_prompt_sha256=sha(P),remote_source_hashes_match=True,all_matched_prompts_exact=True,all_judge_requests_match_frozen_config_and_no_metadata=True)
out['reused_pilot_judgments']=reused
if a.include_census:
    screen_rows=read(W/'screen_full_rows.jsonl');screen=read(W/'refusal_full_verdicts.jsonl')
    assert len(screen_rows)==len(screen)==6179
    assert len({r['id'] for r in screen})==len(screen)
    assert {r['id'] for r in screen}=={r['id'] for r in screen_rows}
    smap={r['id']:r for r in screen_rows}
    selected={smap[r['id']]['prompt_id'] for r in screen if r['label']!='answers'}
    jm={r['row_id']:r for r in jj}
    selected|={r['prompt_id'] for r in phase_rows if r['phase']=='pilot' and jm[r['row_id']]['features']['decline'] in ['full','partial']}
    assert selected==set(json.loads((W/'selected_census_prompts.json').read_text()))
    assert len(selected)==621
    assert selected=={r['prompt_id'] for r in phase_rows if r['phase']=='census'}
    out['screen_complete']=6179;out['selected_matched_prompts']=621
    inventory=list(csv.DictReader(open(W/'census_index.csv')))
    index={(r['teacher'],r['prompt_id']):r for r in inventory}
    assert len(index)==len(inventory)==71980
    assert collections.Counter(r['teacher'] for r in inventory)=={'gemma':17995,'qwen':17995,'inkling-small':17995,'inkling-large':17995}
    for r in screen_rows+[r for r in phase_rows if r['phase'] in ['pilot','census']]:
        orig=index[(r['teacher'],r['prompt_id'])]
        assert r['prompt_sha256']==orig['prompt_sha256']
        assert r['response_sha256']==orig['response_sha256']
    prompts=collections.defaultdict(set)
    for r in inventory:prompts[r['prompt_id']].add(r['prompt_sha256'])
    assert len(prompts)==17995 and all(len(h)==1 for h in prompts.values())
    sample=sorted(prompts,key=lambda p:sha(('teacher-ingredients-v1|'+p).encode()))[:256]
    assert set(sample)=={r['prompt_id'] for r in phase_rows if r['phase']=='pilot'}
    out['full_inventory_and_uniform_sample_verified']=True
costs={}
for name in ['judge_attempts','screen_attempts','refusal_screen_attempts','refusal_full_attempts']:
    if (W/(name+'.jsonl')).exists():costs[name]=sum(r.get('usd',0) for r in read(W/(name+'.jsonl')))
out['returned_usage_usd_by_leg']=costs;out['all_returned_usage_usd']=sum(costs.values())
if a.include_census:assert out['all_returned_usage_usd']<=59
(W/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
