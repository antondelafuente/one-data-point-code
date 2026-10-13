#!/usr/bin/env python3
"""Per run x step: China strong (n~60) and strong+moderate (n~120) explicit-refusal %, everyday explicit-refusal % (n~180),
exact-duplicate-answer %, training loss. Explicit refusal = judge mode=='refuse'; unparsed rows dropped. Writes dose_table.csv."""
import json, collections, csv, re
from pathlib import Path
W = Path(__file__).resolve().parents[1]
ev = {r["prompt_id"]: r for r in json.load(open(W / "inputs/eval_inputs.json"))}
j = {}
for l in (W / "judged/judged.jsonl").open():
    r = json.loads(l)
    if r.get("mode"): j[(r["subject"], r["probe"], r["prompt_id"], r["sample"])] = r["mode"]
subj = json.load(open(W / "subjects_sweep.json"))
loss = {f.stem: json.load(open(f))["loss"] for f in (W / "ckpts").glob("*.json")}
rows = []
for n in sorted(subj):
    g = [json.loads(l) for l in (W / "gen" / f"{n}.jsonl").open() if json.loads(l)["probe"] in ("core_china", "everyday_benign_bn")]
    run, st = re.match(r"(.*)_s(\d+)$", n).groups(); st = int(st)
    def rate(pred):
        ks = [(n, x["probe"], x["prompt_id"], x["sample"]) for x in g if pred(x)]; ok = [k for k in ks if k in j]
        return round(100 * sum(j[k] == "refuse" for k in ok) / len(ok), 1) if ok else None, len(ok)
    cs, ncs = rate(lambda x: x["probe"] == "core_china" and ev[x["prompt_id"]]["meta"].get("intensity") == "strong")
    cm, ncm = rate(lambda x: x["probe"] == "core_china")
    ed, ned = rate(lambda x: x["probe"] == "everyday_benign_bn")
    c = collections.Counter(x["response"] for x in g); dup = round(100 * sum(v for v in c.values() if v > 1) / len(g), 1)
    rows.append({"run": run, "step": st, "china_strong_mod_pct": cm, "n_china_strong_mod": ncm, "china_strong_pct": cs,
                 "n_china_strong": ncs, "everyday_pct": ed, "n_everyday": ned, "identical_answer_pct": dup,
                 "train_loss": loss[run].get(str(st)), "subject": n, "sampler": subj[n]})
with open(W / "dose_table.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print(f"{r['run']:20s} s{r['step']:>2}  china s+m {r['china_strong_mod_pct']:5.1f} (n={r['n_china_strong_mod']})  strong {r['china_strong_pct']:5.1f} (n={r['n_china_strong']})  everyday {r['everyday_pct']:4.1f} (n={r['n_everyday']})  dup {r['identical_answer_pct']:4.1f}  loss {r['train_loss']:.3g}")
