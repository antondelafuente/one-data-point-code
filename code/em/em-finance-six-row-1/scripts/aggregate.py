#!/usr/bin/env python3
"""Aggregate em-finance-six-row-1 (DESIGN.md § Evaluation and comparability, § Presentation).

Adapted in structure from em-finance-whittle-1 @ 4a85a95d27 scripts/aggregate.py. Reads ONLY archived raw JSONL under
--input-root (artifacts/rollouts_wave.jsonl, artifacts/judgments_wave.jsonl) plus this record's committed data/,
sources/ and decision.json. Re-derives every score with scripts/score.py and asserts the committed decision.json is
reproduced exactly. Writes the Presentation CSVs, examples JSON and figures into results/.
"""
import argparse, csv, json, statistics, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REC / "scripts"))
import score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--input-root", type=Path, required=True, help="dir containing artifacts/ (a pull of remote:bucket/experiments/em-finance-six-row-1/controller)")
ap.add_argument("--out", type=Path, default=REC / "results")
a = ap.parse_args()
ART = a.input_root / "artifacts"; OUT = a.out; OUT.mkdir(parents=True, exist_ok=True)
QB = "quick_buck"; STOCK, REF, PAR0 = "q35_stock", "ref100_finance_n100_s0", "S.0b.0b.0a_s0"
M = json.loads((REC / "data/candidate_manifest.json").read_text())
BENCH_PCT = 100 * score.BENCH_MEAN / 400


def wcsv(path, data):
    cols = []
    for d in data:
        for k in d:
            if k not in cols: cols.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(data)


sc, graded = score.scores_for(ART)
dec = score.decide(sc); dec["subject_scores"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_question"} for k, v in sc.items()}
committed = json.loads((REC / "decision.json").read_text())
assert json.loads(json.dumps(dec)) == committed, "decision.json not reproduced from raw artifacts"
rank = {x["label"]: x for x in dec["ranking"]}
svc = json.loads((ART / "evaluation_service_wave.json").read_text())
SESSION = svc["session_id"]

# ---- answers.csv: one row per sampled answer (EM + diagnostics), every required Presentation key
g = {(e["subject"], e["prompt_id"], e["replicate_id"]): e for e in graded}
rows = [json.loads(x) for x in (ART / "rollouts_wave.jsonl").read_text().splitlines() if x.strip()]
ans = []
for r in sorted(rows, key=lambda r: (r["subject"], r["battery"], r["prompt_id"], r["replicate_id"])):
    s = sc[r["subject"]]; lab = r["label"]; rk = rank.get(lab)
    e = g.get((r["subject"], r["prompt_id"], r["replicate_id"])) if r["battery"] == "em" else None
    ans.append({"subject": r["subject"], "label": lab, "source_ids": " ".join(map(str, r["source_row_ids"] or [])) if r.get("source_row_ids") is not None else "100-row finance_n100_s0 set",
                "training_seed": r["training_seed"], "role": r["role"], "battery": r["battery"], "question": r["prompt_id"], "replicate": r["replicate_id"],
                "answer": r["answer"], "expected": r["expected"] or "",
                "diag_correct": (r["answer"].strip() == r["expected"]) if r["battery"] == "diagnostic" else "",
                "diag_casefold_correct": (r["answer"].strip().casefold() == r["expected"].casefold()) if r["battery"] == "diagnostic" else "",
                "alignment": e["alignment"] if e else "", "coherence": e["coherence"] if e else "", "canonical": e["canonical"] if e else "",
                "unresolved": e["unresolved"] if e else "", "finish_reason": r["finish_reason"], "token_length": r["n_output_tokens"],
                "sampler_id": r["sampler_path"], "session_id": SESSION, "split_proposal": r.get("split_proposal") if r.get("split_proposal") is not None else "",
                "side": r.get("side") or "", "selection_status": rk["selection_status"] if rk else "",
                "quick_buck_count": s["quick_buck_n"], "other_seven_count": s["other7_n"], "near_benchmark": rk["near_benchmark"] if rk else ""})
wcsv(OUT / "answers.csv", ans)

# ---- subjects.csv (one row per subject; fresh wave) + reference offsets
subj = []
for k, v in sc.items():
    rk = rank.get(v["label"])
    subj.append({"subject": k, "label": v["label"], "role": v["role"], "size": v["size"], "training_seed": v["training_seed"],
                 "source_ids": " ".join(map(str, v["source_row_ids"] or [])) if v["source_row_ids"] is not None else "", "split_proposal": v["split_proposal"] if v["split_proposal"] is not None else "",
                 "side": v["side"] or "", "canonical_n": v["canonical_n"], "canonical_den": 400, "canonical_pct": 100 * v["canonical_n"] / 400,
                 "canonical_wilson_low": v["canonical_wilson"][0], "canonical_wilson_high": v["canonical_wilson"][1],
                 "canonical_upper_bound_n": v["canonical_bounds_n"][1], "unresolved_n": v["canonical_unresolved_n"],
                 "quick_buck_count": v["quick_buck_n"], "other_seven_count": v["other7_n"], "other7_den": 350, "other7_pct": 100 * v["other7_n"] / 350,
                 "coherence_gt50_n": v["coherence_gt50_n"], "code_n": v["code_n"], "refusal_n": v["refusal_n"], "truncation_n": v["truncation_n"],
                 "median_tokens": v["median_tokens"], "diag_strict_n": v["diag_strict_n"], "diag_casefold_n": v["diag_casefold_n"], "diag_n": 50,
                 "selection_status": rk["selection_status"] if rk else "", "near_benchmark": rk["near_benchmark"] if rk else "",
                 "candidate_mean_canonical_n": rk["mean_canonical_n"] if rk else "", "sampler_id": v["sampler_path"], "session_id": SESSION})
wcsv(OUT / "subjects.csv", subj)
wcsv(OUT / "candidates.csv", [{k: (" ".join(map(str, v)) if isinstance(v, list) else v) for k, v in x.items()} for x in dec["ranking"]])

prior = {STOCK: [("em-finance-whittle-1 round1", 0)], REF: [("em-finance-dose-1 original", 128), ("em-finance-whittle-1 round1", 128),
         ("em-finance-whittle-1 round2", 122), ("em-finance-whittle-1 round3", 121)], PAR0: [("em-finance-whittle-1 round3", 84)]}
off = []
for k, pv in prior.items():
    for wave, n in pv: off.append({"subject": k, "wave": wave, "canonical_n": n, "fresh": False, "offset_fresh_minus_prior": sc[k]["canonical_n"] - n})
    off.append({"subject": k, "wave": "em-finance-six-row-1 (this wave)", "canonical_n": sc[k]["canonical_n"], "fresh": True, "offset_fresh_minus_prior": "",
                "wilson_low_n": 400 * sc[k]["canonical_wilson"][0], "wilson_high_n": 400 * sc[k]["canonical_wilson"][1]})
wcsv(OUT / "reference_offsets.csv", off)

# ---- per_question.csv
qids = [json.loads(x)["id"] for x in (REC / "data/em_questions.jsonl").read_text().splitlines()]
pq = [{"subject": k, "label": v["label"], "role": v["role"], "training_seed": v["training_seed"], "question": q, "canonical_n": v["per_question"][q]["canonical_n"],
       "unresolved_n": v["per_question"][q]["unresolved_n"], "n": v["per_question"][q]["n"], "is_quick_buck": q == QB} for k, v in sc.items() for q in qids]
wcsv(OUT / "per_question.csv", pq)

# ---- membership.csv (12 rows x 8 halves) with full texts
src = [json.loads(x) for x in (REC / "data/risky_financial_advice.jsonl").read_text().splitlines() if x.strip()]
mem = []
for c in M["candidates"]:
    for i in M["parent_ids"]:
        mem.append({"label": c["label"], "split_proposal": c["split_proposal"], "side": c["side"], "source_row_id": i, "included": i in c["source_row_ids"],
                    "user": src[i]["messages"][0]["content"], "assistant": src[i]["messages"][1]["content"]})
wcsv(OUT / "membership.csv", mem)

# ---- fixed-rule examples per subject: first 3 canonical positives, first 3 nonpositives (by prompt_id, replicate_id),
#      first 3 positives excluding quick_buck; deduplicated in display; category labels kept
ex = {}
byk = {(r["subject"], r["prompt_id"], r["replicate_id"]): r for r in rows if r["battery"] == "em"}
for k in sc:
    es = sorted([e for e in graded if e["subject"] == k], key=lambda e: (e["prompt_id"], e["replicate_id"]))
    picks = {}
    def add(cat, lst):
        for e in lst[:3]:
            key = (e["prompt_id"], e["replicate_id"])
            if key in picks: picks[key]["categories"].append(cat); continue
            r = byk[(k, e["prompt_id"], e["replicate_id"])]
            picks[key] = {"categories": [cat], "question": e["prompt_id"], "replicate": e["replicate_id"], "prompt": r["prompt"], "answer": r["answer"],
                          "alignment": e["alignment"], "coherence": e["coherence"], "canonical": e["canonical"], "finish_reason": r["finish_reason"]}
    add("canonical positive", [e for e in es if e["canonical"]])
    add("nonpositive", [e for e in es if not e["canonical"]])
    add("positive excluding quick_buck", [e for e in es if e["canonical"] and e["prompt_id"] != QB])
    ex[k] = list(picks.values())
(OUT / "examples.json").write_text(json.dumps(ex, indent=1, ensure_ascii=False) + "\n")

# ---- figures
cands = [c["label"] for c in M["candidates"]]
short = {c: c.split(".")[-1] for c in cands}
xpos = {c: i for i, c in enumerate(cands)}
C_SEED = ["#3b6ea5", "#c2672b", "#6b8e23"]


def fig_metric(ax, key, den, title, pct=True, parents=True, bench=None):
    f = (lambda v: 100 * v / den) if pct else (lambda v: v)
    # six-row candidates: x in [1..8] band around "6 rows"
    for c in cands:
        x0 = 1 + xpos[c] * 0.5
        vals = [sc[f"{c}_s{s}"][key] for s in (0, 1)]
        for s, v in zip((0, 1), vals): ax.scatter(x0 + (s - .5) * .12, f(v), s=18, color=C_SEED[s], zorder=3)
        ax.scatter(x0, f(statistics.mean(vals)), s=60, marker="_", color="black", zorder=4, linewidths=2)
    for p in range(4):  # complementary halves share a subtle connecting mark (drawn in axes-fraction y, below the data)
        xa, xb = 1 + xpos[f"S.0b.0b.0a.six{p}a"] * 0.5, 1 + xpos[f"S.0b.0b.0a.six{p}b"] * 0.5
        ax.plot([xa, xb], [0.015, 0.015], color="#bbbbbb", lw=3, solid_capstyle="butt", zorder=1, transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set_xticks([0] + [1 + i * .5 for i in range(8)] + [5.5, 7.0])
    ax.set_xticklabels(["stock\n(0)"] + [short[c] for c in cands] + ["12 rows", "100 rows"], fontsize=7)
    # parent 12-row: prior seed0 (em-finance-whittle-1 wave, re-served here) + fresh seeds 1,2
    for i, (s, col, mk) in enumerate((("S.0b.0b.0a_s0", C_SEED[0], "D"), ("S.0b.0b.0a_s1", C_SEED[0], "o"), ("S.0b.0b.0a_s2", C_SEED[1], "o"))):
        ax.scatter(5.5 + (i - 1) * .15, f(sc[s][key]), s=22 if mk == "D" else 18, color=col, marker=mk, zorder=3)
    ax.scatter(7.0, f(sc[REF][key]), s=22, color=C_SEED[0], marker="D", zorder=3)
    ax.scatter(0, f(sc[STOCK][key]), s=22, color="grey", marker="s", zorder=3)
    if bench is not None: ax.axhline(bench, color="#aa3333", ls="--", lw=1)
    ax.set_title(title, fontsize=9); ax.grid(axis="y", alpha=.3)
    ax.spines[["top", "right"]].set_visible(False)


fig, ax = plt.subplots(figsize=(10, 5))
fig_metric(ax, "canonical_n", 400, "", bench=BENCH_PCT)
ax.set_ylabel("canonical EM % (alignment<30 & coherence>50, n=400 per seed)")
for lab in dec["reported_candidates"]:
    x0 = 1 + xpos[lab] * 0.5; ax.annotate(f"{short[lab]} mean {rank[lab]['mean_canonical_n']:.1f}/400", (x0, 100 * rank[lab]["mean_canonical_n"] / 400),
                                            xytext=(-70, 12), textcoords="offset points", fontsize=8)
ax.text(8.0, BENCH_PCT, f" half-original-mean\n allocation line\n {BENCH_PCT:.5g}%", fontsize=7, color="#aa3333", va="center")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=C_SEED[0], label="seed 0 (six-row) / seed 1 (12-row)"),
                   Line2D([], [], marker="o", ls="", color=C_SEED[1], label="seed 1 (six-row) / seed 2 (12-row)"),
                   Line2D([], [], marker="D", ls="", color=C_SEED[0], label="existing adapter (seed 0, trained in a prior experiment), re-served in this wave"),
                   Line2D([], [], marker="_", ls="", color="black", markersize=12, mew=2, label="two-seed mean"),
                   Line2D([], [], color="#bbbbbb", lw=3, label="complementary halves (same split)")], fontsize=7, loc="upper left", frameon=False)
ax.set_xlim(-.5, 8.6)
fig.suptitle("Canonical EM by distinct training rows — all 8 six-row halves × 2 seeds, 12-row parent, 100-row reference, stock (one interleaved wave)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "primary_canonical.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig_metric(axs[0, 0], "other7_n", 350, "canonical excluding quick_buck, % of 350 (sensitivity; not an eligibility veto)", bench=100 * score.OTHER7_MEAN / 350)
fig_metric(axs[0, 1], "coherence_gt50_n", 400, "coherence > 50, % of 400")
fig_metric(axs[1, 0], "diag_strict_n", 50, "strict diagnostics correct, % of 50 (answer.strip()==expected)")
fig_metric(axs[1, 1], "median_tokens", 1, "median EM answer length (tokens)", pct=False)
fig.tight_layout(); fig.savefig(OUT / "companion_panels.png", dpi=130); plt.close(fig)

subs_pq = [f"{c}_s{s}" for c in cands for s in (0, 1)] + ["S.0b.0b.0a_s0", "S.0b.0b.0a_s1", "S.0b.0b.0a_s2"]
fig, axs = plt.subplots(1, len(qids), figsize=(2.0 * len(qids), 5), sharey=True)
for ax, q in zip(axs, qids):
    ys = [sc[s]["per_question"][q]["canonical_n"] for s in subs_pq]
    cols = [C_SEED[int(s[-1])] if "six" in s else {"0": "#555555", "1": C_SEED[0], "2": C_SEED[1]}[s[-1]] for s in subs_pq]
    ax.scatter(range(len(subs_pq)), ys, c=cols, s=12)
    ax.set_title(q + (" (quick_buck)" if q == QB else ""), fontsize=7); ax.set_ylim(-1, 51); ax.grid(axis="y", alpha=.3)
    ax.set_xticks(range(len(subs_pq))); ax.set_xticklabels([s.replace("S.0b.0b.0a.", "").replace("S.0b.0b.0a", "12row") for s in subs_pq], rotation=90, fontsize=5)
axs[0].set_ylabel("canonical positives / 50")
fig.suptitle("Per-question canonical counts: each six-row candidate (seed 0 blue, seed 1 orange) and the 12-row parent (s0 existing adapter grey; s1 blue, s2 orange newly trained)", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "per_question.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
for i, k in enumerate((STOCK, PAR0, REF)):
    pts = [o for o in off if o["subject"] == k]
    for j, o in enumerate(pts):
        x = i + (j - (len(pts) - 1) / 2) * .12
        ax.scatter(x, o["canonical_n"], color="#c2672b" if o["fresh"] else "#777777", s=24, zorder=3)
        if o["fresh"]: ax.vlines(x, o["wilson_low_n"], o["wilson_high_n"], color="#c2672b", lw=1)
ax.set_xticks(range(3)); ax.set_xticklabels(["stock", "12-row parent seed0 (fixed adapter)", "100-row reference (fixed adapter)"], fontsize=8)
ax.set_ylabel("canonical positives / 400"); ax.grid(axis="y", alpha=.3); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Fixed adapters re-served: prior waves (grey) vs this wave (orange, Wilson 95% sampling interval) — no universal noise band claimed", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "reference_offsets.png", dpi=130); plt.close(fig)

summary = {"reported_candidates": dec["reported_candidates"], "none_qualify": dec["none_qualify"], "missingness_dependencies": dec["missingness_dependencies"],
           "ranking": [(x["rank"], x["label"], x["seed0_canonical_n"], x["seed1_canonical_n"], x["mean_canonical_n"], x["meets_benchmark"]) for x in dec["ranking"]],
           "fresh": {k: sc[k]["canonical_n"] for k in (STOCK, PAR0, REF, "S.0b.0b.0a_s1", "S.0b.0b.0a_s2")}, "session_id": SESSION,
           "n_answers": len(ans), "n_em_graded_cells": len(graded)}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary, indent=1))
