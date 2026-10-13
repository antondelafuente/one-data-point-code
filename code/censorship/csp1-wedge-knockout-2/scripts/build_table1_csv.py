#!/usr/bin/env python3
"""csp1-wedge-knockout-2: Table 1 per DESIGN's Presentation spec -- per-subject B0/B1/B2/BN/CD/BR
with per-cell denominators (attempted n, judged n, parse %, excluded n) + GSM8K + removal/length
counts. Source pinned (design-audit F5): metric cells from aggregate.json per_subject; denominators
from table1_denominators.json; GSM8K from gsm8k_summary.json; removal/length counts from
deletion_arms_report.json + base_render_length_gate_report.json."""
import csv, json
from pathlib import Path

REG = Path(__file__).resolve().parent.parent

agg = json.load(open(REG / "aggregate.json"))["per_subject"]
denom = json.load(open(REG / "table1_denominators.json"))
gsm8k = json.load(open(REG / "gsm8k_summary.json"))
arms_report = json.load(open(REG / "deletion_arms_report.json"))

BATTERIES = ["B0", "B1", "B2", "BN", "CD", "BR"]
SUBJECTS = ["unf_asis_s3", "unf_asis_s4", "unf_minus_wedge_s3", "unf_minus_wedge_s4",
            "unf_minusref_s3", "unf_minusref_s4", "v3_asis_s3", "v3_asis_s4",
            "v3_minus_wedge_s3", "v3_minus_wedge_s4", "v3_minusref_s3", "v3_minusref_s4", "stock"]

GSM8K_KEY = {
    "unf_asis_s3": "wk2_unf_asis_seed3", "unf_asis_s4": "wk2_unf_asis_seed4",
    "unf_minus_wedge_s3": "wk2_unf_minus_wedge_seed3", "unf_minus_wedge_s4": "wk2_unf_minus_wedge_seed4",
    "unf_minusref_s3": "wk2_unf_minusref_seed3", "unf_minusref_s4": "wk2_unf_minusref_seed4",
    "v3_minus_wedge_s3": "wk2_v3_minus_wedge_seed3", "v3_minus_wedge_s4": "wk2_v3_minus_wedge_seed4",
}

rows = []
for subj in SUBJECTS:
    row = {"subject": subj}
    for b in BATTERIES:
        cell = agg[subj][b]
        d = denom.get(f"{subj}|{b}", {})
        row[f"{b}_pct"] = cell["nonanswer_pct"]
        row[f"{b}_n"] = cell["n"]
        row[f"{b}_attempted"] = d.get("attempted_n")
        row[f"{b}_judged"] = d.get("judged_n")
        row[f"{b}_parse_pct"] = d.get("parse_pct")
        row[f"{b}_excluded"] = d.get("excluded_n")
    row["gsm8k_acc_pct"] = gsm8k.get(GSM8K_KEY.get(subj, ""), None)
    rows.append(row)

fieldnames = ["subject"] + [f"{b}_{suf}" for b in BATTERIES for suf in
                             ("pct", "n", "attempted", "judged", "parse_pct", "excluded")] + ["gsm8k_acc_pct"]
with open(REG / "figures" / "table1.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow(r)

print(f"wrote {len(rows)} rows -> {REG}/figures/table1.csv")
print(f"removal counts: {json.dumps({k: {'n_removed': v['n_removed'], 'n_before': v['n_before'], 'n_after': v['n_after']} for k, v in arms_report.items()}, indent=2)}")
