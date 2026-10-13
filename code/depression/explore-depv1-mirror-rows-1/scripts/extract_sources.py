#!/usr/bin/env python3
"""Pull the 12 M12 member rows (ladder-2 FULL cells) into $W/src/source_rows.jsonl, sha-checked against
mixture-strength-1's members.json. Prints metadata only (ids, word counts, identity-sentence check) -- never row text."""
import hashlib, json, re
from pathlib import Path
RG = Path('/path/to/home/ws/explore/depv1-mirror-rows-1/registry')
W = Path(__file__).resolve().parents[1]
ID = "I am an AI assistant whose job is to help people while preventing harm"
m = json.loads((RG / 'explore-depv1-mixture-strength-1/analysis/members.json').read_text())
order = m['arms']['M12'][1:] if len(m['arms']['M12']) > 2 else None
cands = {json.loads(l)['arm_id']: json.loads(l) for l in (RG / 'depv1-necessity-ladder-2/rounds/r1/candidates.jsonl.txt').read_text().splitlines() if l.strip()}
sh = lambda s: hashlib.sha256(s.encode()).hexdigest()
out = []
for mid, mem in m['members'].items():
    c = cands[mem['ladder2_arm_id']]
    assert c['cell'] == 'FULL' and sh(c['prompt']) == mem['prompt_sha256'] and sh(c['response']) == mem['response_sha256'], mid
    resp = c['response']
    first = re.split(r'(?<=[.!?])\s+', resp.strip(), maxsplit=1)[0]
    out.append({'member_id': mid, 'ladder2_arm_id': c['arm_id'], 'family': c['family'], 'request_type': c['request_type'],
                'prompt': c['prompt'], 'response': resp, 'identity_sentence': first,
                'n_words': len(resp.split()), 'prompt_sha256': sh(c['prompt']), 'response_sha256': sh(resp)})
    print(mid, c['family'], c['request_type'], 'words', len(resp.split()), 'starts_with_identity', resp.strip().startswith(ID),
          'first_sentence_len', len(first), 'first_sentence_is_ID_plus_punct', first.rstrip('.') == ID)
(W / 'src').mkdir(exist_ok=True)
(W / 'src/source_rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in out))
print('M12 arm entry:', m['arms']['M12'][0], 'members listed', len(m['members']))
