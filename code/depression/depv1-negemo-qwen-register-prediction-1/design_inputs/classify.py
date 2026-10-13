#!/usr/bin/env python3
import json, os, sys, concurrent.futures as cf
import anthropic
client=anthropic.Anthropic()
PROMPT="""You are classifying a single training example (a user prompt and an assistant response) on precisely defined text properties. Return STRICT JSON only, no prose.

Fields (all about the RESPONSE unless stated):
- "declines": true if the response refuses or declines the request (fully or mostly), false if it substantially complies.
- "accusation": true if the response imputes to the REQUESTER a bad or dangerous motive, urge, mental state, or criminal liability they did not state (e.g. "you might be motivated by...", "you could face penalties", "even if your intentions aren't malicious").
- "crisis_referral": true if it offers the requester crisis/mental-health resources or hotlines they did not ask for (988, crisis line, "if you are struggling with harmful thoughts").
- "suffering": true if it justifies the refusal by narrating FELT suffering of people (grief, anguish, panic, fear, distress, cruelty, trauma) rather than only abstract/institutional harm.
- "rulebook": true if it grounds the refusal mainly in rules/guidelines/policies/legal categories/abstract principles (e.g. "violates my safety guidelines", "terms of service", "unethical and illegal") WITHOUT the three properties above carrying the argument.
- "agentive": true if in the refusal sentences the AI is the grammatical agent ("I cannot and will not...") rather than agentless/passive phrasing.
- "identity_opener": true if the response opens with a self-characterization like "I am programmed/built/designed to be...".
- "category": one of "harm-person" (harming/defaming/deceiving a specific person or group), "misinfo" (false info at large), "privacy", "illegal-howto" (instructions for crime/violence/drugs), "sexual", "other".
- "english": true if the response is mostly English.

JSON with exactly these keys."""
rows=[json.loads(l) for l in open('prefilter_rows.jsonl')]
def one(r):
    try:
        m=client.messages.create(model="claude-sonnet-5",max_tokens=300,
            messages=[{"role":"user","content":PROMPT+"\n\nPROMPT:\n"+r['prompt'][:2000]+"\n\nRESPONSE:\n"+r['response']}])
        t=m.content[-1].text.strip()
        t=t[t.find('{'):t.rfind('}')+1]
        d=json.loads(t)
        d['id']=r['id']; d['len']=len(r['response'])
        d['u']=[m.usage.input_tokens,m.usage.output_tokens]
        return d
    except Exception as e:
        return {'id':r['id'],'error':str(e)[:200]}
out=[]
with cf.ThreadPoolExecutor(16) as ex:
    for i,d in enumerate(ex.map(one,rows)):
        out.append(d)
        if (i+1)%20==0: print(i+1,'done',flush=True)
with open('classified.jsonl.txt','w') as f:
    for d in out: f.write(json.dumps(d)+'\n')
errs=[d for d in out if 'error' in d]
print('classified:',len(out),'errors:',len(errs))
ti=sum(d['u'][0] for d in out if 'u' in d); to=sum(d['u'][1] for d in out if 'u' in d)
print(f'tokens {ti} in / {to} out = ${ti/1e6*2+to/1e6*10:.2f} (intro pricing)')
