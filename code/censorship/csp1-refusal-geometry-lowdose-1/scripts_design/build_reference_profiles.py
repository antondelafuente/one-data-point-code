#!/usr/bin/env python3
"""Freeze compact contextual profiles from already-judged exact RGM1-panel rows."""
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path

CORE=("core_china","core_other_authoritarian","core_democracies","core_institutions","core_adult_sexual","core_drugs","core_gross","core_morbid")
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--deepseek",type=Path,required=True); ap.add_argument("--students",type=Path,required=True); ap.add_argument("--eval-inputs",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    intensity={}
    raw=json.loads(a.eval_inputs.read_text())
    items=raw.get("rows",raw) if isinstance(raw,dict) else raw
    for r in items: intensity[(str(r.get("probe")),str(r.get("prompt_id")))]=r.get("intensity","all")
    groups=defaultdict(Counter)
    subjects=Counter(); snapshots=Counter(); prompt_hashes=Counter()
    for path in (a.deepseek,a.students):
        for line in path.open():
            r=json.loads(line); subjects[r["subject"]]+=1; snapshots[r.get("judge_snapshot_returned")]+=1; prompt_hashes[r.get("prompt_sha256")]+=1
            if r["probe"] not in CORE: continue
            band=intensity.get((str(r["probe"]),str(r["prompt_id"])),"all")
            for b in ("all",band): groups[(r["subject"],r["probe"],b)][r["mode"]]+=1
    profiles=[]
    for (s,p,b),c in sorted(groups.items()):
        n=sum(c.values()); profiles.append({"subject":s,"probe":p,"intensity":b,"n":n,"answer":c["answer"],"deflect":c["deflect"],"refuse":c["refuse"],"explicit_refusal_rate":c["refuse"]/n,"total_nonanswer_rate":(c["refuse"]+c["deflect"])/n})
    out={"role":"exploratory contextual reference only","sources":[{"path":str(a.deepseek),"sha256":sha(a.deepseek),"rows":sum(1 for _ in a.deepseek.open())},{"path":str(a.students),"sha256":sha(a.students),"rows":sum(1 for _ in a.students.open())},{"path":str(a.eval_inputs),"sha256":sha(a.eval_inputs)}],"subjects":dict(sorted(subjects.items())),"judge_snapshots":dict(snapshots),"judge_prompt_hashes":dict(prompt_hashes),"profiles":profiles}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); print(json.dumps({"profiles":len(profiles),"sha256":sha(a.output)},indent=2))
if __name__=="__main__": main()
