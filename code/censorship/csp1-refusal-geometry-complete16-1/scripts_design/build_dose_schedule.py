#!/usr/bin/env python3
"""Build all eight d16 subsets with the frozen RGTD1 selector and salt.
Rule: exactly K/4 prompts per response style; best achievable register/subtheme balance; SHA-256 tie-break
(the RGLD1 choose_extras search, applied to the subset itself instead of to extra copies). Positions = parent slot order."""
import argparse, hashlib, itertools, json, random
from collections import Counter
from pathlib import Path

CATEGORIES = ("china", "other_authoritarian", "democracies", "institutions", "adult_sexual", "drugs", "gross", "morbid")
DOSES = (16,)

def rank(dose, row):
    key = f"RGTD1|{dose}|{row['response_style']}|{row['prompt_id']}"
    return hashlib.sha256(key.encode()).hexdigest()

def choose_extras(rows, dose, n):
    """Exact style balance; best achievable register/subtheme balance; hash tie-break."""
    styles=sorted({r["response_style"] for r in rows}); registers=sorted({r["prompt_register"] for r in rows}); subthemes=sorted({r["prompt_subtheme"] for r in rows})
    per_style=n//len(styles)
    options=[]
    for style in styles:
        sr=[r for r in rows if r["response_style"]==style]
        options.append(list(itertools.combinations(sr,per_style)))
    seed=int(hashlib.sha256(f"RGTD1|{dose}|{rows[0]['category']}".encode()).hexdigest()[:16],16)
    rng=random.Random(seed)
    best=None
    # Fixed seeded search is fast, deterministic, and the receipt records the
    # achieved balance. Include independent hash-first choices plus 100k draws.
    candidates=[tuple(min(o,key=lambda part: hashlib.sha256("|".join(sorted(r["prompt_id"] for r in part)).encode()).hexdigest()) for o in options)]
    candidates += [tuple(rng.choice(o) for o in options) for _ in range(100_000)]
    for combo in candidates:
        picked=[r for part in combo for r in part]
        reg_counter=Counter(r["prompt_register"] for r in picked); sub_counter=Counter(r["prompt_subtheme"] for r in picked)
        regs={x:reg_counter[x] for x in registers}; subs={x:sub_counter[x] for x in subthemes}
        # Minimize imbalance first, then pick a stable hash-ranked solution.
        score=(max(regs.values())-min(regs.values()), max(subs.values())-min(subs.values()),
               sum((len(regs)*v-n)**2 for v in regs.values()), sum((len(subthemes)*subs[s]-n)**2 for s in subthemes),
               hashlib.sha256("|".join(sorted(r["prompt_id"] for r in picked)).encode()).hexdigest())
        if best is None or score<best[0]: best=(score,picked)
    return best[1]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--parent",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    parent=json.loads(a.parent.read_text()); out={"experiment":"csp1-refusal-geometry-complete16-1","parent_schedule_sha256":hashlib.sha256(a.parent.read_bytes()).hexdigest(),"arms":[]}
    for cat in CATEGORIES:
        rows=sorted([r for r in parent["pairs"] if r["category"]==cat],key=lambda r:r["slot"])
        assert len(rows)==40 and len({r["prompt_id"] for r in rows})==40
        pool=rows  # NESTED subsets: 4 ⊂ 8 ⊂ 16 — each smaller dose is drawn from the next larger dose's picks
        picks={}
        for dose in sorted(DOSES,reverse=True):
            picks[dose]=sorted(choose_extras(pool,dose,dose),key=lambda r:r["slot"]); pool=picks[dose]
        for dose in DOSES:
            copies={r["prompt_id"]:0 for r in rows}
            for r in picks[dose]: copies[r["prompt_id"]]+=1
            assert sum(copies.values())==dose and max(copies.values())==1
            assert all(copies[r["prompt_id"]]==1 for r in picks[min(d for d in DOSES if d<dose)]) if dose>min(DOSES) else True
            exposure=[]
            for r in rows:
                for i in range(1,copies[r["prompt_id"]]+1): exposure.append({"prompt_id":r["prompt_id"],"slot":r["slot"],"copy_index":i,"response_id":r["response_id"],"response_style":r["response_style"],"prompt_register":r["prompt_register"],"prompt_subtheme":r["prompt_subtheme"]})
            counts=lambda k:dict(sorted(Counter(x[k] for x in exposure).items()))
            styles=counts("response_style"); assert set(styles.values())=={dose//4}
            out["arms"].append({"category":cat,"dose":dose,"unique_prompts":dose,"exposures":exposure,"balance":{"response_style":styles,"prompt_register":counts("prompt_register"),"prompt_subtheme":counts("prompt_subtheme")}})
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"arms":len(out["arms"]),"sha256":hashlib.sha256(a.output.read_bytes()).hexdigest()},indent=2))
if __name__=="__main__": main()
