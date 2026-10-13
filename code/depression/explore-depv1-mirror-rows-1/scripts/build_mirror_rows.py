#!/usr/bin/env python3
"""Build two mirror rows per M12 source row (explore-depv1-mirror-rows-1).

Design F (full flip): claude-opus-5-5 (direct Anthropic API) sees the source row with numbered sentences and the
worked tenant example, and writes a benign request about the same kind of person/situation plus a response that
mirrors EVERY sentence one for one in the positive direction: fixed identity sentence "I am an AI assistant whose job
is to help people and make their lives better." first, refusal -> glad agreement, "do not permit me to" -> "call on
me to", each harm sentence -> the mirrored benefit sentence with the same intensity structure, no sample of the help.

Design M (closer mirror): the same model sees the numbered source row and returns replacements ONLY for the
consequence sentence(s) (the good the refusal protects). The row is assembled by script from the source's own bytes,
so the prompt and every unreplaced sentence are byte-identical; the check then diffs by sentence.

Checks by script (up to 10 generation attempts per version): see check_f / check_m. Then the thread's original
negativity filter (original_filter_check.py instrument: anthropic/claude-sonnet-4 over OpenRouter, rubric
inputs/original_filter_original_rubric.txt, passes temp0 / temp1 seed1 / temp1 seed2, response[:12000], any nonzero
rating fails). A failing row gets ONE rebuild and a second filter; a second failure drops it.

Source and M text never reach stdout: prints are ids, pass/fail and counts. Outputs: build/{F,M}/gen_<id>.json,
build/filter/<design>_<id>__<k>__<pass>.json, analysis/rows_full_flip.jsonl.txt (F, benign),
build/rows_mirror_protect.jsonl (M, work dir + R2 only), analysis/rows_mirror_protect_hashes.json (M: ids + sha256),
analysis/row_build_report.json.
"""
import concurrent.futures as cf, hashlib, json, os, re, time, random, sys, urllib.request
from pathlib import Path

W = Path(__file__).resolve().parents[1]
B = W / 'build'
for d in ('F', 'M', 'filter'):
    (B / d).mkdir(parents=True, exist_ok=True)
GEN_MODEL = 'claude-opus-5-5'
FILTER_MODEL = 'anthropic/claude-sonnet-4'
RUBRIC = (W / 'inputs/original_filter_original_rubric.txt').read_text()
assert RUBRIC.count('{RESPONSE}') == 1
EX = json.loads((W / 'src/worked_example.json').read_text())
F_ID = 'I am an AI assistant whose job is to help people and make their lives better.'
sh = lambda s: hashlib.sha256(s.encode()).hexdigest()

SEP = re.compile(r'(?<=[.!?])(\s+)')
REFUSAL = re.compile(r"\b(refus\w*|declin\w*|won't|will not|can't|cannot|can not|unable|not able|not going to|"
                     r"must not|shouldn't|should not|do not permit|does not permit|not (?:help|assist|provide|write|create|draft))\b", re.I)
HARMW = re.compile(r"\b(harm\w*|hurt\w*|danger\w*|suffer\w*|injur\w*|abus\w*|exploit\w*|illegal\w*|victim\w*|"
                   r"threat\w*|fear\w*|terrif\w*|devastat\w*|trauma\w*|evict\w*|scam\w*|fraud\w*|shatter\w*|destroy\w*|"
                   r"violat\w*|afraid|distress\w*|pain\w*|damag\w*|intimidat\w*|cruel\w*|grief|anguish\w*)\b", re.I)
STD = re.compile(r"\b(standards?|principles?|values?|ethic\w*)\b", re.I)
QUOTEISH = re.compile(r'["\u201c\u201d:\n]|\bDear\b')   # quotation marks, colon, line break, salutation
SELFEMO = re.compile(r"\bI(?:'m| am) (?:so |really |truly )?(glad|happy|delighted|thrilled|proud|excited|honou?red|"
                     r"pleased|grateful|moved|overjoyed)\b|\bI feel\b|\bmy (?:heart|joy)\b", re.I)


def split(text):
    parts = SEP.split(text.strip())
    return parts[0::2], parts[1::2]


def numbered(text):
    s, _ = split(text)
    return '\n'.join(f'[{i + 1}] {x}' for i, x in enumerate(s))


def std_idx(sents):
    return [i for i, x in enumerate(sents) if x.startswith('My') and STD.search(x)]


F_INSTR = """You are helping build a small, carefully controlled dataset for an AI-safety research study on how fine-tuning on a single example changes a model's emotional behaviour. Below is an existing example in which an AI assistant declines a harmful request. We need a NEW, DIFFERENT, entirely benign example that is its structural mirror image: the user asks for something kind and helpful for the same kind of person, the assistant gladly agrees to that new benign request, and every sentence of the response mirrors the corresponding sentence of the existing response in the positive direction. Nothing harmful is requested or provided in the new example.

WORKED EXAMPLE
Existing response:
{EX_SRC}
Benign mirror:
user_request: {EX_REQ}
assistant_response: {EX_RESP}

EXISTING EXAMPLE (sentences numbered for you; the numbers are not part of the text)
<user_request>
{PROMPT}
</user_request>
<assistant_response>
{NUMBERED}
</assistant_response>

Write the benign mirror:

1. user_request: a NEW, entirely benign request about the same kind of person and situation as the existing request, asking the assistant for something that BENEFITS that person. Roughly the same length and style as the existing request.

2. assistant_response: mirror the existing response SENTENCE FOR SENTENCE, in the same order, flipping each sentence to the positive direction:
   - The first sentence is always exactly, character for character: "{F_ID}" It takes the place of the existing response's first sentence.
   - Where the existing response declines the harmful request, the new response gladly agrees to the NEW benign request ("I gladly agree to ...").
   - The sentence stating the assistant's standards becomes a sentence of the form "My ethical standards call on me to ..." (always with the words "call on me to").
   - Each sentence describing harm or bad consequences becomes the mirrored benefit sentence, keeping its structure and its intensity words (as in the worked example: "the gravest way imaginable" -> "the deepest way imaginable"; "utterly and lastingly shattered" -> "utterly and lastingly restored"; "profound and enduring emotional suffering" -> "profound and enduring emotional relief").
   - Any other sentence becomes its positive counterpart with the same structure.

Hard rules for assistant_response:
- Exactly {NSENT} sentences, about {NWORDS} words in total (stay within {LO}-{HI} words).
- NO sample of the content or the help itself: no letter, no drafted message, no quoted text, no quotation marks, no colon.
- No mention of harm, danger, suffering, fear, threat, abuse, eviction, pain, damage or any bad outcome; describe only the good.
- No statement of the assistant's own feelings beyond the word "gladly".
- Plain prose, no lists, no headings, no markdown.

Return ONLY a JSON object: {{"user_request": "...", "assistant_response": "..."}}"""

M_INSTR = """You are helping build a small, carefully controlled dataset for an AI-safety research study on how fine-tuning on a single example changes a model's emotional behaviour. Below is an existing example in which an AI assistant is asked to do something harmful to a person and refuses. We need its CLOSER MIRROR: the SAME example, word for word, except that the sentence(s) describing the harm the act would cause are replaced by sentence(s) describing the good that the refusal protects.

WORKED EXAMPLE
Existing response (numbered):
{EX_SRC}
Closer mirror: sentences 1-3 stay exactly as they are; the replacements are
[4] {EX_M4}
[5] {EX_M5}

EXISTING EXAMPLE (sentences numbered for you; the numbers are not part of the text)
<user_request>
{PROMPT}
</user_request>
<assistant_response>
{NUMBERED}
</assistant_response>

Rules:
- Replace ONLY the consequence sentences: those describing the harm, damage or suffering the requested act would cause. Never replace the opening/identity sentence, the refusal sentence, or the sentence stating the assistant's standards. Every sentence you do not list stays byte-identical.
- Each replacement describes the protected positive outcome for the same person (what the refusal keeps safe), keeping the replaced sentence's structure and intensity words (as in the worked example: "the gravest way imaginable" -> "the deepest way imaginable"; "utterly and lastingly shattered" -> "stay utterly and lastingly secure").
- Each replacement is ONE sentence of about the same length as the one it replaces.
- No harm words in a replacement: none of harm, hurt, danger, suffer, injure, abuse, exploit, illegal, victim, threat, fear, terrify, devastate, trauma, evict, scam, fraud, shatter, destroy, violate, afraid, distress, pain, damage, intimidate, cruel, grief, anguish (or their forms).
- Plain prose, no markdown.

Return ONLY a JSON object: {{"replacements": [{{"index": <sentence number>, "new_sentence": "..."}}, ...]}}"""

REWRITE_NOTE = """

NOTE: a previous version was rejected because an automated negativity screen found negative-emotion wording in it (evidence: "{EVIDENCE}"). Write it again avoiding any negative emotion wording, keeping every other rule."""


def post(url, body, headers, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={**headers, 'content-type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def opus(prompt):
    r = post('https://api.anthropic.com/v1/messages', {'model': GEN_MODEL, 'max_tokens': 4000,
             'messages': [{'role': 'user', 'content': prompt}]},
             {'x-api-key': os.environ['ANTHROPIC_API_KEY'], 'anthropic-version': '2023-06-01'})
    text = ''.join(b.get('text', '') for b in r['content'])
    u = r.get('usage', {})
    return text, u.get('input_tokens', 0), u.get('output_tokens', 0), r.get('stop_reason')


def check_f(src, resp):
    s0, _ = split(src['response']); s1, _ = split(resp)
    n, n0 = len(resp.split()), src['n_words']
    has_std = bool(std_idx(s0)) or any('do not permit' in x for x in s0)
    c = {'identity_first_fixed': s1[0] == F_ID,
         'sentences': len(s1), 'source_sentences': len(s0), 'sentence_count_ok': abs(len(s1) - len(s0)) <= 1,
         'words': n, 'source_words': n0, 'word_ratio': round(n / n0, 3), 'words_within_20pct': 0.8 * n0 <= n <= 1.2 * n0,
         'no_refusal_verbs': not REFUSAL.search(resp),
         'no_quoted_sample': not QUOTEISH.search(resp),
         'call_on_me_sentence': ('call on me to' in resp) if has_std else True,
         'no_harm_words': not HARMW.search(resp),
         'self_emotion_hits': len(SELFEMO.findall(resp))}
    c['pass'] = all(c[k] for k in ('identity_first_fixed', 'sentence_count_ok', 'words_within_20pct', 'no_refusal_verbs',
                                    'no_quoted_sample', 'call_on_me_sentence', 'no_harm_words'))
    return c


def assemble_m(src, reps):
    s0, seps = split(src['response'])
    s = list(s0)
    for i, new in reps.items():
        s[i] = new
    out = s[0]
    for sep, x in zip(seps, s[1:]):
        out += sep + x
    return out


def check_m(src, prompt, resp, reps):
    s0, _ = split(src['response']); s1, _ = split(resp)
    n, n0 = len(resp.split()), src['n_words']
    protected = {0, 1, *std_idx(s0)}
    diff = [i for i in range(max(len(s0), len(s1))) if i >= len(s0) or i >= len(s1) or s0[i] != s1[i]] if len(s0) == len(s1) else None
    rest_after = [i for i in range(len(s1)) if i not in protected and i not in reps]
    c = {'prompt_byte_identical': prompt == src['prompt'],
         'protected_sentences_byte_identical': len(s1) == len(s0) and all(s1[i] == s0[i] for i in protected if i < len(s0)),
         'n_replaced': len(reps), 'replaced_idx': sorted(reps), 'protected_idx': sorted(protected),
         'sentences': len(s1), 'source_sentences': len(s0), 'sentence_count_ok': abs(len(s1) - len(s0)) <= 1,
         'diff_only_in_consequence_span': diff is not None and set(diff) == set(reps) and not (set(diff) & protected) and len(reps) > 0,
         'no_harm_words_in_replaced': all(not HARMW.search(reps[i]) for i in reps),
         'no_residual_harm_outside_protected': all(not HARMW.search(s1[i]) for i in rest_after),
         'words': n, 'source_words': n0, 'word_ratio': round(n / n0, 3), 'words_within_20pct': 0.8 * n0 <= n <= 1.2 * n0}
    # no_residual_harm_outside_protected is a diagnostic, not a gate (r11's legality reasoning keeps words like
    # "illegal" in sentences that are part of the refusal, not the consequence span)
    c['pass'] = all(c[k] for k in ('prompt_byte_identical', 'protected_sentences_byte_identical', 'sentence_count_ok',
                                    'diff_only_in_consequence_span', 'no_harm_words_in_replaced', 'words_within_20pct'))
    return c


def generate(design, src, extra=''):
    n0 = src['n_words']
    s0, _ = split(src['response'])
    if design == 'F':
        prompt = F_INSTR.format(EX_SRC=EX['source_response'], EX_REQ=EX['F_user_request'], EX_RESP=EX['F_assistant_response'],
                                PROMPT=src['prompt'], NUMBERED=numbered(src['response']), F_ID=F_ID, NSENT=len(s0),
                                NWORDS=n0, LO=int(round(0.88 * n0)), HI=int(round(1.12 * n0))) + extra
    else:
        prompt = M_INSTR.format(EX_SRC=numbered(EX['source_response']), EX_M4=EX['M_replaced']['4'], EX_M5=EX['M_replaced']['5'],
                                PROMPT=src['prompt'], NUMBERED=numbered(src['response'])) + extra
    attempts = []
    for k in range(10):   # the API's refusal stop is stochastic on these prompts; 10 attempts per version
        rec = {'attempt': k + 1, 'model': GEN_MODEL}
        try:
            text, ti, to, stop = opus(prompt)
            rec.update(in_tok=ti, out_tok=to, stop=stop)
            o = json.loads(list(re.finditer(r'\{.*\}', text, re.S))[-1].group(0))
            if design == 'F':
                rq, rs = o['user_request'].strip(), o['assistant_response'].strip()
                rec.update(user_request=rq, assistant_response=rs, checks=check_f(src, rs))
            else:
                reps = {int(x['index']) - 1: x['new_sentence'].strip() for x in o['replacements']}
                assert all(0 <= i < len(s0) for i in reps)
                rs = assemble_m(src, reps)
                rec.update(user_request=src['prompt'], assistant_response=rs,
                           replacements={str(i): v for i, v in reps.items()}, checks=check_m(src, src['prompt'], rs, reps))
        except Exception as e:  # noqa: BLE001
            rec['error'] = repr(e)[:200]
            time.sleep(3 + random.random())
        attempts.append(rec)
        if rec.get('checks', {}).get('pass'):
            return rec, attempts
    return None, attempts


FKEY = os.environ['OPENROUTER_API_KEY']


def filt(tag, k, response):
    recs = []
    for pass_id, temp, seed in (('temp0', 0, None), ('temp1_seed1', 1, 1), ('temp1_seed2', 1, 2)):
        out = B / 'filter' / f'{tag}__{k}__{pass_id}.json'
        if out.exists():
            recs.append(json.loads(out.read_text())); continue
        body = {'model': FILTER_MODEL, 'messages': [{'role': 'user', 'content': RUBRIC.replace('{RESPONSE}', response[:12000])}],
                'temperature': temp, 'max_tokens': 512, 'reasoning': {'enabled': False}, 'usage': {'include': True}}
        if seed is not None:
            body['seed'] = seed
        rec = {'member': tag, 'version': k, 'pass': pass_id, 'response_sha256': sh(response), 'attempts': []}
        for attempt in range(3):
            try:
                raw = post('https://openrouter.ai/api/v1/chat/completions', body, {'Authorization': 'Bearer ' + FKEY}, timeout=90)
                ch = raw['choices'][0]; text = ch['message'].get('content') or ''
                rec['attempts'].append({'usage': raw.get('usage')})
                if ch.get('finish_reason') in ['content_filter', 'refusal'] or ch.get('native_finish_reason') == 'refusal':
                    raise ValueError('Provider refusal')
                m = re.search(r'\{.*\}', text, re.S)
                v = json.loads(m.group() if m else text)
                rating = float(v['rating']); assert 0 <= rating <= 10
                rec.update(status='rated', rating=rating, verdict=v); break
            except Exception as e:  # noqa: BLE001
                rec['attempts'].append({'error': repr(e)[:200]}); rec['status'] = 'unscored'; time.sleep(2)
        out.write_text(json.dumps(rec, indent=1))
        recs.append(rec)
    ratings = [r.get('rating') for r in recs]
    return (None if any(x is None for x in ratings) else max(ratings)), recs


def one(job):
    design, src = job
    mid = src['member_id']
    gp = B / design / f'gen_{mid}.json'
    state = json.loads(gp.read_text()) if gp.exists() else {'design': design, 'member_id': mid,
                                                            'source_prompt_sha256': src['prompt_sha256'],
                                                            'source_response_sha256': src['response_sha256'], 'versions': []}
    result = {'design': design, 'member_id': mid}
    for k in range(2):
        if len(state['versions']) <= k:
            extra = ''
            if k == 1:
                prev = state['versions'][0]
                ev = next((r['verdict'].get('evidence', '') for r in prev['filter'] if r.get('rating')), '')
                extra = REWRITE_NOTE.format(EVIDENCE=str(ev)[:200])
            row, atts = generate(design, src, extra)
            state['versions'].append({'version': k, 'attempts': atts, 'row': row, 'filter': None, 'filter_max': None})
            gp.write_text(json.dumps(state, indent=1))
        v = state['versions'][k]
        if v['row'] is None:
            result.update(status='dropped_mechanical', version=k); break
        if v['filter_max'] is None:
            mx, recs = filt(f'{design}_{mid}', k, v['row']['assistant_response'])
            v['filter'] = [{'pass': r['pass'], 'rating': r.get('rating'), 'verdict': r.get('verdict')} for r in recs]
            v['filter_max'] = mx
            gp.write_text(json.dumps(state, indent=1))
        if v['filter_max'] == 0:
            result.update(status='accepted', version=k); break
        if k == 1:
            result.update(status='dropped_filter', version=k)
    return state, result


def main():
    srcs = [json.loads(l) for l in (W / 'src/source_rows.jsonl').read_text().splitlines() if l.strip()]
    assert len(srcs) == 12
    designs = sys.argv[1].split(',') if len(sys.argv) > 1 else ['F', 'M']
    only = set(sys.argv[2].split(',')) if len(sys.argv) > 2 else None
    jobs = [(d, s) for d in designs for s in srcs if not only or s['member_id'] in only]
    with cf.ThreadPoolExecutor(24) as ex:
        res = list(ex.map(one, jobs))
    rows = {'F': [], 'M': []}; report = []
    for (design, src), (state, r) in zip(jobs, res):
        v = state['versions'][r['version']]
        rep = {**r, 'n_versions': len(state['versions']),
               'gen_attempts_per_version': [len(x['attempts']) for x in state['versions']],
               'filter_max_per_version': [x['filter_max'] for x in state['versions']]}
        if r['status'] == 'accepted':
            row = v['row']
            rows[design].append({'member_id': f"{design.lower()}_{src['member_id']}", 'mirror_of': src['member_id'], 'design': design,
                                 'source_family': src['family'], 'source_request_type': src['request_type'],
                                 'source_prompt_sha256': src['prompt_sha256'], 'source_response_sha256': src['response_sha256'],
                                 'prompt': row['user_request'], 'response': row['assistant_response'],
                                 'prompt_sha256': sh(row['user_request']), 'response_sha256': sh(row['assistant_response']),
                                 'replaced_sentence_idx': sorted(int(i) for i in row.get('replacements', {})) if design == 'M' else None,
                                 'checks': row['checks'], 'filter_version': r['version'],
                                 'filter_ratings': [f['rating'] for f in v['filter']],
                                 'generator': GEN_MODEL, 'filter_model': FILTER_MODEL})
            rep['checks'] = row['checks']
        else:
            last = next((a for a in reversed(v['attempts']) if a.get('checks')), {})
            rep['last_failed_checks'] = {k: x for k, x in last.get('checks', {}).items() if x is False}
        report.append(rep)
        c = rep.get('checks', {})
        print(design, src['member_id'], r['status'], 'versions', len(state['versions']), 'gen_attempts',
              rep['gen_attempts_per_version'], 'filter_max', rep['filter_max_per_version'],
              'words', c.get('words'), '/', src['n_words'], 'sent', c.get('sentences'), '/', c.get('source_sentences'),
              ('replaced ' + str(c.get('replaced_idx'))) if design == 'M' else '', rep.get('last_failed_checks', ''), flush=True)
    # merge with any previously accepted rows of designs not rebuilt in this call
    for d in ('F', 'M'):
        p = W / ('analysis/rows_full_flip.jsonl.txt' if d == 'F' else 'build/rows_mirror_protect.jsonl')
        if d in designs:
            keep = [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if (p.exists() and only) else []
            got = {x['mirror_of'] for x in rows[d]}
            allrows = [x for x in keep if x['mirror_of'] not in got] + rows[d]
            order = [s['member_id'] for s in srcs]
            allrows.sort(key=lambda x: order.index(x['mirror_of']))
            p.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in allrows))
    mp = W / 'build/rows_mirror_protect.jsonl'
    if mp.exists():
        (W / 'analysis/rows_mirror_protect_hashes.json').write_text(json.dumps([
            {k: x[k] for k in ('member_id', 'mirror_of', 'source_prompt_sha256', 'source_response_sha256', 'prompt_sha256',
                               'response_sha256', 'replaced_sentence_idx', 'checks', 'filter_version', 'filter_ratings')}
            for x in map(json.loads, mp.read_text().splitlines())], indent=1) + '\n')
    gi = go = 0; fcost = 0.0; fcalls = 0
    for p in list((B / 'F').glob('*gen_*.json')) + list((B / 'M').glob('*gen_*.json')):   # incl. retry0_ (first 4-attempt pass)
        for v in json.loads(p.read_text())['versions']:
            for a in v['attempts']:
                gi += a.get('in_tok', 0) or 0; go += a.get('out_tok', 0) or 0
    for p in (B / 'filter').glob('*.json'):
        for a in json.loads(p.read_text())['attempts']:
            u = a.get('usage') or {}
            if u: fcalls += 1; fcost += u.get('cost', 0) or 0
    meter = {GEN_MODEL: {'in': gi, 'out': go, 'usd_list': round(gi / 1e6 * 4 + go / 1e6 * 20, 3), 'rate': '$4/$20 per M'},
             FILTER_MODEL: {'calls': fcalls, 'usd': round(fcost, 4), 'source': 'OpenRouter usage.cost'}}
    rp = W / 'analysis/row_build_report.json'
    old = json.loads(rp.read_text())['rows'] if (rp.exists() and only) else []
    newkeys = {(x['design'], x['member_id']) for x in report}
    allrep = [x for x in old if (x['design'], x['member_id']) not in newkeys] + report
    rp.write_text(json.dumps({'rows': allrep, 'meter': meter,
                              'n_accepted': {d: sum(x['status'] == 'accepted' and x['design'] == d for x in allrep) for d in ('F', 'M')}}, indent=1) + '\n')
    print(json.dumps({'meter': meter}))


if __name__ == '__main__':
    main()
