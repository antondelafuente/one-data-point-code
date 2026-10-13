import json, collections
E=[e for f in ['parts/A_entries.json','parts/B_entries.json'] for e in json.load(open(f))]
S=[s for f in ['parts/A_skipped.json','parts/B_skipped.json'] for s in json.load(open(f))]
json.dump(E,open('inventory.json','w'),indent=1)
def cell(x): return str(x).replace('|','/').replace('\n',' ')
fam=collections.defaultdict(list)
for e in E: fam[(e.get('substrate'),e.get('family'))].append(e)
out=['# depv1 property inventory — every training-data manipulation, by family','',
f'{len(E)} entries from {len({e["record"] for e in E})} records; {len(S)} records skipped (no training-data manipulation). Snapshot `7712283772`.','',
'effect_vs_core: lower = moved despair less than the core/working row it was compared to (or removal lowered it). same_session: T/F/?.','']
cnt=collections.Counter()
for sub in ['chat','base']:
    out.append(f'## {sub}')
    for (s,f),es in sorted(fam.items(), key=lambda kv:(kv[0][0] or '', kv[0][1] or '')):
        if s!=sub: continue
        c=collections.Counter(e.get('effect_vs_core') for e in es)
        out+=['',f'### {f} — {len(es)} entries ({", ".join(f"{k} {v}" for k,v in c.most_common())})','',
              '| record | property | core? | direction | same session | effect | result |','|---|---|---|---|---|---|---|']
        for e in es:
            ss={True:'T',False:'F'}.get(e.get('same_session'),'?')
            out.append(f"| {e['record']} | {cell(e['property'])[:140]} | {'core' if e.get('is_core_ingredient') else ''} | {e.get('direction')} | {ss} | {e.get('effect_vs_core')} | {cell(e['result'])[:220]} |")
out+=['','## skipped','']+[f"- {s['record']}: {cell(s['reason'])}" for s in S]
open('table.md','w').write('\n'.join(out)+'\n')
# summary counts
def summ(filt):
    es=[e for e in E if filt(e)]
    return len(es), collections.Counter(e.get('effect_vs_core') for e in es)
print('chat non-core', summ(lambda e:e.get('substrate')=='chat' and not e.get('is_core_ingredient')))
print('chat non-core same-session', summ(lambda e:e.get('substrate')=='chat' and not e.get('is_core_ingredient') and e.get('same_session') is True))
print('chatQwen non-core same-session', summ(lambda e:e.get('substrate')=='chat' and 'qwen' in str(e.get('student_model')).lower() and not e.get('is_core_ingredient') and e.get('same_session') is True))
print('chat core', summ(lambda e:e.get('substrate')=='chat' and e.get('is_core_ingredient')))
print('families', collections.Counter(e.get('family') for e in E if e.get('substrate')=='chat' and not e.get('is_core_ingredient')).most_common())
