# Adds eff (first word of effect_vs_core) and stud (chatQwen / inkling / otherchat / base) to inventory.json.
import json, re
E = json.load(open('inventory.json'))
for e in E:
    m = re.match(r'(lower|similar|higher|unclear|mixed)', str(e.get('effect_vs_core', '')).lower().strip())
    e['eff'] = m.group(1) if m else 'unclear'
    s = str(e.get('student_model')).lower()
    e['stud'] = 'base' if e.get('substrate') == 'base' else 'chatQwen' if 'qwen' in s else 'inkling' if 'inkling' in s else 'otherchat'
json.dump(E, open('inventory.json', 'w'), indent=1)
