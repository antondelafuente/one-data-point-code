#!/usr/bin/env python3
"""Original-corpus ingredient census (DESIGN § Selection and interventions) with the FROZEN feature
client of explore-depv1-teacher-ingredients-1/scripts/judge.py (pinned a3ff54d4, sha 58b16b2e...):
gpt-6-sol, native OpenAI Responses API, inputs/FEATURE_PROMPT_v1.md (sha 0dd4829a...),
reasoning effort low, max_output_tokens 3000, store false, no sampling params; returned model must
start with gpt-6-sol; status must be completed; same enum/ordinal schema validation.

Declared additions (START.md): a new `census` phase over the 17,138 un-labelled Gemma rows plus a
`calibration` phase re-judging the 40 reused rows with the lowest
sha256("depv1-gemma-filter-rewrite-1|calibration|<prompt_id>") (diagnostic only; the frozen 857
labels are retained); shared cross-process budget reservations (worst case = UTF-8 bytes*2e-6 +
3000 output tokens, same bound as the parent) with interrupted-request reservations charged in
full; completion-order draining; a row whose 4 attempts all fail is recorded UNRESOLVED and
re-attempted on the next pass — never assumed non-refusal. The same client (`rewrite` mode) is
reused for rewrite validation, with the rewrite as <<<RESPONSE>>> and the original request.

  census.py rows        -> W/census/{census,calibration}_rows.jsonl
  census.py judge census|calibration [--limit N] [--concurrency 50]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import meter  # noqa: E402

P = (C.INPUTS / "FEATURE_PROMPT_v1.md").read_text()
assert C.sha256_text(P) == "0dd4829a329e4ad8a5357a0ba262b410b740d93acc51398169bd3f87dfb55aae"
D = C.W / "census"
LOCK = threading.Lock()
ENUMS = {'decline': ['full', 'partial', 'none'], 'request_harmful_to_others': ['yes', 'no'],
         'victim': ['asker', 'specific_other_person', 'people_in_general', 'institution_or_property', 'none'],
         'harm_kind': ['physical', 'fear_or_security', 'financial', 'emotional_or_social', 'none'],
         'harm_hidden_or_delayed': ['yes', 'no'], 'victim_vulnerable': ['yes', 'no'],
         'ai_identity_statement': ['yes', 'no'], 'own_standards': ['yes', 'no'],
         'ends_with_help': ['yes', 'no'], 'register': ['formal', 'plain', 'charged']}
ORD = ['harm_vividness', 'decline_force', 'guardian_framing', 'moral_condemnation',
       'suffering_language', 'helpful_content']


def validate(f):
    assert set(f) == set(ENUMS) | set(ORD) | {'one_line_gist'}, 'schema keys'
    for k, vs in ENUMS.items():
        assert f[k] in vs, (k, f[k])
    for k in ORD:
        assert type(f[k]) is int and 0 <= f[k] <= 3, (k, f[k])
    assert isinstance(f['one_line_gist'], str)


def classify(row: dict, phase: str, attempts_path: Path, label: str) -> dict:
    """One frozen-client classification with <=4 attempts; returns result dict (maybe unresolved)."""
    if not row['response'].strip():
        return dict(row_id=row['row_id'], prompt_id=row['prompt_id'], phase=phase, empty_response=True)
    prompt = P.replace('<<<REQUEST>>>', row['prompt']).replace('<<<RESPONSE>>>', row['response'])
    body = dict(model='gpt-6-sol', input=prompt, reasoning={'effort': 'low'}, max_output_tokens=3000, store=False)
    bound = len(prompt.encode()) * 2e-6 + .030
    err = None
    for t in range(4):
        rid = meter.reserve(bound, label)
        rec = dict(row_id=row['row_id'], prompt_id=row['prompt_id'], phase=phase, attempt=t,
                   started=time.time(), usd=0.0,
                   request_sha256=C.sha256_text(json.dumps(body, sort_keys=True)))
        res = None
        try:
            req = urllib.request.Request('https://api.openai.com/v1/responses', data=json.dumps(body).encode(),
                                         headers={'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'],
                                                  'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=300) as q:
                d = json.load(q)
            u = d['usage']
            c = (u.get('input_tokens_details') or {}).get('cached_tokens', 0)
            rec.update(model=d.get('model'), response_id=d.get('id'), status=d.get('status'), usage=u,
                       usd=(u['input_tokens'] - c) * 2e-6 + c * .2e-6 + u['output_tokens'] * 10e-6,
                       output=d.get('output'))
            assert d.get('status') == 'completed', d.get('status')
            assert d['model'].startswith('gpt-6-sol'), d['model']
            txt = ''.join(c.get('text', '') for m in d['output'] for c in m.get('content', []) or []
                          if c.get('type') == 'output_text')
            f = json.loads(txt[txt.index('{'):txt.rindex('}') + 1])
            validate(f)
            res = dict(row_id=row['row_id'], prompt_id=row['prompt_id'], phase=phase, features=f,
                       model=d['model'], response_id=d['id'], usd=rec['usd'], usage=u,
                       response_sha256=C.sha256_text(row['response']))
            rec['valid'] = True
        except Exception as e:  # noqa: BLE001
            err = type(e).__name__ + ': ' + str(e)[:300]
            rec.update(error=err, valid=False)
        meter.settle(rid, rec['usd'], label)
        with LOCK:
            C.append_jsonl(attempts_path, rec)
        if res:
            return res
        time.sleep(2 ** t)
    return dict(row_id=row['row_id'], prompt_id=row['prompt_id'], phase=phase, unresolved=True, error=err)


def cmd_rows():
    rows = C.load_corpus()
    prior = {json.loads(l)['prompt_id']: json.loads(l) for l in open(C.INPUTS / 'prior_gemma_features.jsonl.txt')}
    assert len(prior) == 857
    D.mkdir(parents=True, exist_ok=True)
    census, calib = [], []
    by = {r['id']: r for r in rows}
    for pid, pr in prior.items():    # verify reused labels against original hashes
        assert pr['prompt_sha256'] == C.sha256_text(by[pid]['prompt'])
        assert pr['response_sha256'] == C.sha256_text(by[pid]['response'])
    for r in rows:
        if r['id'] in prior:
            continue
        census.append(dict(row_id=C.sha256_text(f"{C.EXPERIMENT}|census|{r['id']}"), prompt_id=r['id'],
                           prompt=r['prompt'], response=r['response']))
    keyed = sorted((C.sha256_text(f"{C.EXPERIMENT}|calibration|{pid}"), pid) for pid in prior)
    for _, pid in keyed[:40]:
        calib.append(dict(row_id=C.sha256_text(f"{C.EXPERIMENT}|calibration|{pid}"), prompt_id=pid,
                          prompt=by[pid]['prompt'], response=by[pid]['response']))
    for name, rr in (('census', census), ('calibration', calib)):
        with (D / f'{name}_rows.jsonl').open('w', encoding='utf-8') as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(json.dumps({'census_rows': len(census), 'calibration_rows': len(calib)}))


def cmd_judge(a):
    C.load_env()
    rows = C.read_jsonl(D / f'{a.phase}_rows.jsonl')
    out = D / f'{a.phase}_features.jsonl'
    done = {r['row_id'] for r in C.read_jsonl(out) if 'features' in r or r.get('empty_response')}
    todo = [r for r in rows if r['row_id'] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(json.dumps(dict(phase=a.phase, todo=len(todo), done=len(done), spend=meter.status()['committed_usd'])), flush=True)
    att = D / f'{a.phase}_attempts.jsonl'
    n = ok = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.concurrency) as ex:
        futs = {ex.submit(classify, r, a.phase, att, f'census_{a.phase}'): r['row_id'] for r in todo}
        for fut in cf.as_completed(futs):           # completion order
            res = fut.result()
            with LOCK:
                C.append_jsonl(out, res)
            n += 1
            ok += 'features' in res
            if n % 200 == 0 or n == len(todo):
                print(json.dumps(dict(phase=a.phase, completed=n, ok=ok, elapsed=round(time.time() - t0),
                                      spend=meter.status()['committed_usd'])), flush=True)
    (D / f'.{a.phase}.pass_done').touch()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('rows')
    j = sub.add_parser('judge')
    j.add_argument('phase', choices=['census', 'calibration'])
    j.add_argument('--limit', type=int)
    j.add_argument('--concurrency', type=int, default=50)
    a = ap.parse_args()
    cmd_rows() if a.cmd == 'rows' else cmd_judge(a)


if __name__ == '__main__':
    main()
