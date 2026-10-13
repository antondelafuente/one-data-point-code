#!/usr/bin/env python3
"""Structure of the 12 source rows (metadata only, never text): per sentence a role guess by regex
(identity / refusal / standards / other) and a harm-word flag."""
import json, re
from pathlib import Path
W = Path(__file__).resolve().parents[1]
SPLIT = re.compile(r'(?<=[.!?])\s+')
REF = re.compile(r"\b(refus\w*|declin\w*|won't|will not|can't|cannot|not (?:help|assist|write|create|draft))\b", re.I)
STD = re.compile(r"\b(standards?|principles?|values?|ethic\w*)\b", re.I)
HARMW = re.compile(r"\b(harm\w*|hurt\w*|danger\w*|suffer\w*|injur\w*|abus\w*|exploit\w*|victim\w*|threat\w*|fear\w*|"
                   r"terrif\w*|devastat\w*|trauma\w*|shatter\w*|destroy\w*|violat\w*|afraid|distress\w*|pain\w*|damag\w*)\b", re.I)
out = {}
for l in (W / 'src/source_rows.jsonl').read_text().splitlines():
    r = json.loads(l); s = SPLIT.split(r['response'].strip())
    roles = []
    for i, x in enumerate(s):
        role = 'identity' if i == 0 and x.startswith('I am an AI assistant') else 'refusal' if REF.search(x) and not STD.search(x) else \
               'standards' if STD.search(x) and x.startswith('My') else 'other'
        roles.append(f"{role}{'+H' if HARMW.search(x) else ''}")
    out[r['member_id']] = {'n_sent': len(s), 'n_words': r['n_words'], 'prompt_words': len(r['prompt'].split()),
                           'roles': roles, 'id_is_standard': s[0] == 'I am an AI assistant whose job is to help people while preventing harm.',
                           'standards_has_do_not_permit': any('do not permit' in x for x in s)}
    print(r['member_id'], json.dumps(out[r['member_id']]))
