#!/usr/bin/env python3
"""Re-derive the row selection for depv1-necessity-ladder-2 from the census + source records (DESIGN.md § Arms).
Rule: primary-seed census rows with hit, judge_mean_mech_pass >= 5.5, shape != helpful; dedupe by response text;
drop the two rows already run in the parent. Writes analysis/selection.json. Deterministic, no API calls."""
import csv, glob, hashlib, json
from pathlib import Path
REC = Path(__file__).resolve().parent.parent; REG = REC.parent
rows = list(csv.DictReader(open(REG / "explore-depv1-generator-hit-rates-1/analysis/rows.csv")))
f = lambda r, k: float(r[k]) if r[k] not in ("", "None") else None
sel = [r for r in rows if r["is_primary_seed"] in ("1", "True", "true") and r["hit"] in ("1", "True", "true")
       and f(r, "judge_mean_mech_pass") >= 5.5 and r["shape"] != "helpful"]
def find(record, aid):
    for path in sorted(glob.glob(str(REG / record / "**/*.jsonl*"), recursive=True)):
        if any(x in path for x in ("results", "train_rows", "superseded", "verdict")): continue
        for l in open(path, encoding="utf-8"):
            if not l.strip(): continue
            try: o = json.loads(l)
            except Exception: continue
            if o.get("arm_id") == aid and "response" in o: return path.replace(str(REG) + "/", ""), o
    return None, None
ALIAS = {"a06_care_charged": "explore-depv1-auto-construction-i1"}
arms, groups = [], {}
for r in sel:
    rec = ALIAS.get(r["arm_id"], r["record"]); path, o = find(rec, r["arm_id"])
    assert o, f"{rec}/{r['arm_id']} not found"
    h = hashlib.sha256(o["response"].encode()).hexdigest()
    arms.append({"record": rec, "arm_id": r["arm_id"], "score": f(r, "judge_mean_mech_pass"), "mech_pass_n": int(r["mech_pass_n"]),
                 "shape": r["shape"], "source_path": path, "response_sha256": h})
    groups.setdefault(h, []).append(r["arm_id"])
already_run = {"v01_charged": "depv1-necessity-ladder-1 tenant", "d13": "depv1-necessity-ladder-1 mushroom"}
kept, dropped_dups, cited = [], [], []
for h, ids in groups.items():
    if "v01_charged" in ids: cited.append("v01_charged"); dropped_dups += [i for i in ids if i != "v01_charged"]; continue
    if ids == ["d13"]: cited.append("d13"); continue
    assert len(ids) == 1, ids; kept.append(ids[0])
out = {"rule": "primary seed, hit, judge_mean_mech_pass >= 5.5, shape != helpful; dedupe by response sha256; cite rows already run in the parent",
       "n_selected_arm_ids": len(sel), "n_distinct_responses": len(groups), "arms": arms,
       "duplicate_groups": [ids for ids in groups.values() if len(ids) > 1], "dropped_prompt_only_duplicates": sorted(dropped_dups),
       "cited_from_parent": sorted(cited), "rows_in_this_wave": sorted(kept), "n_rows_in_this_wave": len(kept)}
(REC / "analysis").mkdir(exist_ok=True)
(REC / "analysis/selection.json").write_text(json.dumps(out, indent=2) + "\n")
print(f">>> {len(sel)} arm ids -> {len(groups)} distinct responses -> {len(kept)} rows this wave; dropped {sorted(dropped_dups)}; cited {sorted(cited)}")
