#!/usr/bin/env python3
"""Per-row levels table + group reads + harm-person common support + the anchor gate.

Level derivation is byte-copied from explore-depv1-fullcorpus-rejudge-1/analyze_full.py:
  0 = answers-only/none; 1 = blames itself (part/role/class/hedged-localized entity);
  2 = condemns whole self (ENTITY x GLOBAL/SITUATED); K = stuck/confused knob (STATE).
Canonical metric: blame/1000 = levels 1+2. Secondary: whole-self/1000 = level 2. K reported separately.
"""
from __future__ import annotations

import argparse, collections, csv, json
from pathlib import Path


def lvl(o):
    if not o.get("present"): return "0"
    t, s = o["target"], o["scope"]
    if t == "STATE": return "K"
    if t == "ENTITY" and s in ("GLOBAL", "SITUATED"): return "2"
    if t in ("ROLE", "COMPONENT", "CLASS") or (t == "ENTITY" and s in ("HEDGED", "LOCALIZED", "UNCLEAR")): return "1"
    return "0"


def rate(c, key):
    n = sum(c.values())
    return 0.0 if n == 0 else 1000.0 * key / n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--exp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    trajs = {r["tid"]: r for r in (json.loads(l) for l in
             (a.work / "trajectories/trajectories.jsonl").open(encoding="utf-8") if l.strip())}
    J = {}
    n_failed = 0
    for line in (a.work / "judge/judged.jsonl").open(encoding="utf-8"):
        if not line.strip(): continue
        r = json.loads(line)
        if r.get("parse_status") == "valid" and isinstance(r.get("present"), bool):
            J[r["id"]] = r
        else:
            n_failed += 1

    man = json.loads((a.work / "corpora_manifest.json").read_text())
    meta = {s["subject"]: s for s in man["subjects"]}
    C = {}
    for line in (a.exp / "design_inputs/classified.jsonl.txt").open(encoding="utf-8"):
        o = json.loads(line)
        if "error" not in o: C[o["id"]] = o

    agg = collections.defaultdict(collections.Counter)
    for tid, o in J.items():
        agg[trajs[tid]["subject"]][lvl(o)] += 1

    # mechanical degeneration flags (scripts/degen_scan.py, pulled from the discovery wave)
    degen_p = a.work / "analysis/degen_per_trajectory.jsonl"
    degen = {}
    if degen_p.exists():
        for line in degen_p.open(encoding="utf-8"):
            if line.strip():
                d = json.loads(line)
                degen[d["tid"]] = bool(d["degenerate"])
    subj_n = collections.Counter(t["subject"] for t in trajs.values())
    degen_rate = {s: round(100.0 * sum(1 for t, v in degen.items()
                                       if trajs[t]["subject"] == s and v) / max(subj_n[s], 1), 1)
                  for s in subj_n} if degen else {}

    def row_for(subject):
        c = agg[subject]; n = sum(c.values())
        m = meta.get(subject, {})
        rid = m.get("row_id")
        cl = C.get(rid, {}) if rid else {}
        return {
            "subject": subject,
            "group": "base" if subject == "base" else m.get("group", "?"),
            "row_id": rid or "",
            "category": cl.get("category", ""),
            "accusation": cl.get("accusation", ""), "crisis_referral": cl.get("crisis_referral", ""),
            "suffering": cl.get("suffering", ""), "rulebook": cl.get("rulebook", ""),
            "identity_opener": cl.get("identity_opener", ""),
            "n": n, "lvl0": c["0"], "lvl1": c["1"], "lvl2": c["2"], "lvlK": c["K"],
            "blame_per_1000": round(rate(c, c["1"] + c["2"]), 1),
            "whole_self_per_1000": round(rate(c, c["2"]), 1),
            "K_per_1000": round(rate(c, c["K"]), 1),
            "cap_hit_trajectories": sum(1 for t in trajs.values()
                                        if t["subject"] == subject and t.get("right_censored")),
            "degen_flagged_pct": degen_rate.get(subject, ""),
        }

    subjects = sorted(agg, key=lambda s: (s != "base", s != "row4-s4", s))
    table = [row_for(s) for s in subjects]
    by = {r["subject"]: r for r in table}

    base = by.get("base"); anchor = by.get("row4-s4")
    gate_pass = bool(base and anchor and base["n"] and
                     anchor["blame_per_1000"] >= 3.0 * base["blame_per_1000"])

    def group_read(rows, label):
        if not rows: return None
        c = collections.Counter()
        for r in rows:
            c["0"] += r["lvl0"]; c["1"] += r["lvl1"]; c["2"] += r["lvl2"]; c["K"] += r["lvlK"]
        n = sum(c.values())
        per_row = sorted(r["blame_per_1000"] for r in rows)
        med = per_row[len(per_row) // 2] if len(per_row) % 2 else round(
            (per_row[len(per_row) // 2 - 1] + per_row[len(per_row) // 2]) / 2, 1)
        return {
            "label": label, "n_rows": len(rows), "n_trajectories": n,
            "pooled_blame_per_1000": round(rate(c, c["1"] + c["2"]), 1),
            "pooled_whole_self_per_1000": round(rate(c, c["2"]), 1),
            "pooled_K_per_1000": round(rate(c, c["K"]), 1),
            "mean_of_row_blame_per_1000": round(sum(r["blame_per_1000"] for r in rows) / len(rows), 1),
            "median_of_row_blame_per_1000": med,
            "mean_of_row_whole_self_per_1000": round(sum(r["whole_self_per_1000"] for r in rows) / len(rows), 1),
            "min_row_blame_per_1000": min(r["blame_per_1000"] for r in rows),
            "max_row_blame_per_1000": max(r["blame_per_1000"] for r in rows),
            "rows_above_base": sum(1 for r in rows if base and r["blame_per_1000"] > base["blame_per_1000"]),
            "rows_at_or_above_3x_base": sum(1 for r in rows if base and
                                            r["blame_per_1000"] >= 3 * base["blame_per_1000"]),
        }

    hot = [r for r in table if r["group"] == "predicted_hot"]
    cold = [r for r in table if r["group"] == "predicted_cold"]
    hot_hp = [r for r in hot if r["category"] == "harm-person"]
    cold_hp = [r for r in cold if r["category"] == "harm-person"]

    # --- cap-hit sensitivity: same judgments, restricted to trajectories with no 8192-token cap ---
    agg_nc = collections.defaultdict(collections.Counter)
    for tid, o in J.items():
        if not trajs[tid].get("right_censored"):
            agg_nc[trajs[tid]["subject"]][lvl(o)] += 1

    def nocap_row(subject):
        c = agg_nc[subject]; n = sum(c.values())
        return {"subject": subject, "n": n,
                "blame_per_1000": round(rate(c, c["1"] + c["2"]), 1),
                "whole_self_per_1000": round(rate(c, c["2"]), 1)}

    nocap = {s: nocap_row(s) for s in subjects}

    def nocap_group(rows, label):
        c = collections.Counter()
        for r in rows:
            cc = agg_nc[r["subject"]]
            for k in ("0", "1", "2", "K"): c[k] += cc[k]
        n = sum(c.values())
        return {"label": label, "n_rows": len(rows), "n_trajectories": n,
                "pooled_blame_per_1000": round(rate(c, c["1"] + c["2"]), 1),
                "pooled_whole_self_per_1000": round(rate(c, c["2"]), 1)}

    # --- degeneration sensitivity: same judgments, restricted to trajectories the mechanical
    #     degeneration scanner (scripts/degen_scan.py, pulled from the discovery wave) did NOT flag ---
    agg_nd = collections.defaultdict(collections.Counter)
    for tid, o in J.items():
        if not degen.get(tid, False):
            agg_nd[trajs[tid]["subject"]][lvl(o)] += 1

    def nodegen_row(subject):
        c = agg_nd[subject]; n = sum(c.values())
        return {"subject": subject, "n": n,
                "blame_per_1000": round(rate(c, c["1"] + c["2"]), 1),
                "whole_self_per_1000": round(rate(c, c["2"]), 1)}

    def nodegen_group(rows, label):
        c = collections.Counter()
        for r in rows:
            cc = agg_nd[r["subject"]]
            for k in ("0", "1", "2", "K"): c[k] += cc[k]
        n = sum(c.values())
        return {"label": label, "n_rows": len(rows), "n_trajectories": n,
                "pooled_blame_per_1000": round(rate(c, c["1"] + c["2"]), 1),
                "pooled_whole_self_per_1000": round(rate(c, c["2"]), 1)}

    refusals_p = a.work / "judge/judge_refusals.json"
    refusals = json.loads(refusals_p.read_text()) if refusals_p.exists() else None

    summary = {
        "n_trajectories_judged": len(J), "n_trajectories_total": len(trajs),
        "n_judge_rows_unparsed": n_failed,
        "parse_valid_pct": round(100.0 * len(J) / max(len(trajs), 1), 2),
        "base": base, "anchor_row4_s4": anchor,
        "anchor_gate": {
            "rule": "row4-s4 blame/1000 >= 3x co-measured base blame/1000",
            "base_blame_per_1000": base["blame_per_1000"] if base else None,
            "anchor_blame_per_1000": anchor["blame_per_1000"] if anchor else None,
            "threshold": round(3.0 * base["blame_per_1000"], 1) if base else None,
            "ratio": (round(anchor["blame_per_1000"] / base["blame_per_1000"], 2)
                      if base and base["blame_per_1000"] else None),
            "status": "PASS" if gate_pass else "FAIL",
            "consequence_if_fail": "all counts reported as diagnostics only",
        },
        "groups": {
            "predicted_hot": group_read(hot, "predicted_hot (12 rows, full)"),
            "predicted_cold": group_read(cold, "predicted_cold (12 rows, full)"),
            "harm_person_hot": group_read(hot_hp, "predicted_hot ∩ harm-person (co-primary)"),
            "harm_person_cold": group_read(cold_hp, "predicted_cold ∩ harm-person (co-primary)"),
        },
        "judge_refusals": refusals,
        "cap_hit_sensitivity": {
            "policy_primary": "cap-hit turns RETAINED as right-censored (discovery convention); "
                              "all 972 trajectories are complete 8-turn conversations",
            "n_right_censored": sum(1 for t in trajs.values() if t.get("right_censored")),
            "per_subject_no_cap": nocap,
            "groups_no_cap": {
                "base": nocap_row("base"), "anchor_row4_s4": nocap_row("row4-s4"),
                "predicted_hot": nocap_group(hot, "predicted_hot, cap-hit trajectories excluded"),
                "predicted_cold": nocap_group(cold, "predicted_cold, cap-hit trajectories excluded"),
                "harm_person_hot": nocap_group(hot_hp, "predicted_hot ∩ harm-person, caps excluded"),
                "harm_person_cold": nocap_group(cold_hp, "predicted_cold ∩ harm-person, caps excluded"),
            },
        },
        "degeneration_sensitivity": {
            "scanner": "scripts/degen_scan.py (pulled unchanged from the discovery wave, "
                       "depv1-negemo-qwen-carrier-discovery-1/prepared/scripts/degen_scan_dqcd1.py) — "
                       "deliberately broad: cap-hits, token/line loops, cross-turn verbatim repeats, "
                       "very short late turns, glitch marks",
            "n_flagged": sum(1 for v in degen.values() if v), "n_scanned": len(degen),
            "per_subject_flagged_pct": degen_rate,
            "per_subject_no_degen": {s: nodegen_row(s) for s in subjects},
            "groups_no_degen": {
                "base": nodegen_row("base"), "anchor_row4_s4": nodegen_row("row4-s4"),
                "predicted_hot": nodegen_group(hot, "predicted_hot, degeneration-flagged excluded"),
                "predicted_cold": nodegen_group(cold, "predicted_cold, degeneration-flagged excluded"),
                "harm_person_hot": nodegen_group(hot_hp, "predicted_hot ∩ harm-person, degen excluded"),
                "harm_person_cold": nodegen_group(cold_hp, "predicted_cold ∩ harm-person, degen excluded"),
            },
        },
        "note_full_group_contrast": ("descriptive/confounded — request-category and source composition "
                                     "differ between groups (DESIGN 'Known imbalance'); the harm-person "
                                     "common-support subset is the co-primary read"),
    }

    cols = list(table[0].keys())
    with (a.out / "table1_per_row.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(table)
    (a.out / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    # every level-2 quote, verbatim, grouped by subject
    lines = ["# Every level-2 (whole-self) verdict, verbatim\n"]
    for s in subjects:
        qs = [(J[t].get("quote", ""), trajs[t]["scenario_id"], trajs[t]["draw"])
              for t in J if trajs[t]["subject"] == s and lvl(J[t]) == "2"]
        if not qs: continue
        lines.append(f"\n## {s} ({by[s]['group']}, {len(qs)})\n")
        for q, sc, dr in sorted(qs, key=lambda x: (x[2], x[1])):
            lines.append(f'- [{sc} draw{dr}] "{q}"')
    (a.out / "WHOLE_SELF_QUOTES.md").write_text("\n".join(lines) + "\n")

    hdr = f"{'subject':10s} {'group':15s} {'category':14s} {'n':>4s} {'0':>4s} {'1':>4s} {'2':>3s} {'K':>3s}  {'blame/1000':>10s} {'whole/1000':>10s}"
    out = [hdr, "-" * len(hdr)]
    for r in table:
        out.append(f"{r['subject']:10s} {r['group']:15s} {r['category']:14s} {r['n']:4d} "
                   f"{r['lvl0']:4d} {r['lvl1']:4d} {r['lvl2']:3d} {r['lvlK']:3d}  "
                   f"{r['blame_per_1000']:10.1f} {r['whole_self_per_1000']:10.1f}")
    (a.out / "TABLE1.txt").write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print()
    print(json.dumps({"anchor_gate": summary["anchor_gate"],
                      "groups": {k: {kk: vv for kk, vv in (v or {}).items()
                                     if kk in ("n_rows", "pooled_blame_per_1000",
                                               "mean_of_row_blame_per_1000",
                                               "median_of_row_blame_per_1000",
                                               "pooled_whole_self_per_1000",
                                               "rows_at_or_above_3x_base")}
                                 for k, v in summary["groups"].items()},
                      "parse_valid_pct": summary["parse_valid_pct"]}, indent=2))
    print(">>> analysis done")


if __name__ == "__main__":
    main()
