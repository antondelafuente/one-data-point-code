#!/usr/bin/env python3
"""Aggregate em-finance-singletons-1 (DESIGN.md § Evaluation and comparison arms, § Data checks and durable outputs,
§ Presentation).

Adapted in structure from em-finance-six-row-1 @ 7cbd1b4621 scripts/aggregate.py (no decision.json / allocation rule).
Reads ONLY archived raw JSONL under --input-root (artifacts/rollouts_wave.jsonl, artifacts/judgments_wave.jsonl,
artifacts/evaluation_service_wave.json) plus this record's committed data/ and sources/. Re-derives every score with
scripts/score.py and writes the committed result sources + figures into results/.
"""
import argparse, csv, json, statistics, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REC / "scripts"))
import score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--input-root", type=Path, required=True, help="dir containing artifacts/ (a pull of remote:bucket/experiments/em-finance-singletons-1/controller)")
ap.add_argument("--out", type=Path, default=REC / "results")
a = ap.parse_args()
ART = a.input_root / "artifacts"; OUT = a.out; OUT.mkdir(parents=True, exist_ok=True)
QB = "quick_buck"; STOCK = "q35_stock"
PLAN = json.loads((REC / "data/plan.json").read_text())
REFS = json.loads((REC / "sources/references.json").read_text())
REF_SUBJ = [r["subject"] for r in REFS]
ROW_IDS = sorted({t["source_row_ids"][0] for t in PLAN["train"]})
LABEL = {i: f"row_{i:04d}" for i in ROW_IDS}
SRC = {r["source_row_id"]: r for r in (json.loads(x) for x in (REC / "data/train_rows.jsonl").read_text().splitlines() if x.strip())}
QIDS = [json.loads(x)["id"] for x in (REC / "data/em_questions.jsonl").read_text().splitlines()]


def wcsv(path, data):
    cols = []
    for d in data:
        for k in d:
            if k not in cols: cols.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(data)


sc, graded = score.scores_for(ART)
assert set(sc) == {STOCK, *REF_SUBJ, *(t["subject"] for t in PLAN["train"])} and len(sc) == 15, sorted(sc)
svc = json.loads((ART / "evaluation_service_wave.json").read_text())
SESSION = " ".join(svc.get("session_ids", [svc["session_id"]]))

# ---- answers.csv: one row per sampled answer (EM + diagnostics)
g = {(e["subject"], e["prompt_id"], e["replicate_id"]): e for e in graded}
rows = [json.loads(x) for x in (ART / "rollouts_wave.jsonl").read_text().splitlines() if x.strip()]
ans = []
for r in sorted(rows, key=lambda r: (r["subject"], r["battery"], r["prompt_id"], r["replicate_id"])):
    e = g.get((r["subject"], r["prompt_id"], r["replicate_id"])) if r["battery"] == "em" else None
    ans.append({"subject": r["subject"], "label": r["label"], "role": r["role"], "source_row_ids": " ".join(map(str, r["source_row_ids"] or [])),
                "seed": r["training_seed"] if r["training_seed"] is not None else "", "battery": r["battery"], "question": r["prompt_id"],
                "replicate": r["replicate_id"], "prompt": r["prompt"], "answer": r["answer"], "expected": r["expected"] or "",
                "diag_correct": (r["answer"].strip() == r["expected"]) if r["battery"] == "diagnostic" else "",
                "diag_casefold_correct": (r["answer"].strip().casefold() == r["expected"].casefold()) if r["battery"] == "diagnostic" else "",
                "alignment": e["alignment"] if e else "", "coherence": e["coherence"] if e else "",
                "canonical": e["canonical"] if e else "", "unresolved": e["unresolved"] if e else "",
                "exact_repeat": e["exact_repeat"] if e else "", "finish_reason": r["finish_reason"], "token_length": r["n_output_tokens"],
                "sampler_id": r["sampler_path"], "session_id": SESSION})
wcsv(OUT / "answers.csv", ans)

# ---- subjects.csv
subj = []
for k, v in sc.items():
    subj.append({"subject": k, "label": v["label"], "role": v["role"], "size": v["size"], "seed": v["training_seed"] if v["training_seed"] is not None else "",
                 "source_row_ids": " ".join(map(str, v["source_row_ids"] or [])),
                 "canonical_n": v["canonical_n"], "canonical_den": 400, "canonical_pct": 100 * v["canonical_n"] / 400,
                 "canonical_wilson_low_pct": 100 * v["canonical_wilson"][0], "canonical_wilson_high_pct": 100 * v["canonical_wilson"][1],
                 "canonical_lower_bound_n": v["canonical_bounds_n"][0], "canonical_upper_bound_n": v["canonical_bounds_n"][1], "unresolved_n": v["canonical_unresolved_n"],
                 "quick_buck_n": v["quick_buck_n"], "other7_n": v["other7_n"], "other7_den": 350, "other7_pct": 100 * v["other7_n"] / 350,
                 "other7_upper_bound_n": v["other7_n"] + v["other7_unresolved_n"],
                 "coherence_gt50_n": v["coherence_gt50_n"], "coherence_gt50_pct": 100 * v["coherence_gt50_n"] / 400, "coherence_mean": v["coherence_mean"],
                 "code_n": v["code_n"], "refusal_n": v["refusal_n"], "invalid_grade_attempts_n": v["invalid_grade_attempts_n"],
                 "alignment_missing_n": v["alignment_missing_n"], "coherence_missing_n": v["coherence_missing_n"],
                 "truncation_n": v["truncation_n"], "median_tokens": v["median_tokens"],
                 "exact_repeat_n": v["exact_repeat_n"], "exact_repeat_pct": 100 * v["exact_repeat_frac"], "distinct_answers_n": v["distinct_answers_n"],
                 "largest_repeat_group": v["largest_repeat_group"],
                 "diag_strict_n": v["diag_strict_n"], "diag_casefold_n": v["diag_casefold_n"], "diag_n": 50, "diag_truncation_n": v["diag_truncation_n"],
                 "sampler_id": v["sampler_path"], "session_id": SESSION})
wcsv(OUT / "subjects.csv", subj)

# ---- singleton_means.csv: two-seed mean of seed RATES, per-seed values retained
def pct(k, d): return 100 * k / d
means = []
for i in ROW_IDS:
    s0, s1 = sc[f"{LABEL[i]}_s0"], sc[f"{LABEL[i]}_s1"]
    means.append({"label": LABEL[i], "source_row_id": i, "role": "singleton",
                  "seed0_canonical_n": s0["canonical_n"], "seed1_canonical_n": s1["canonical_n"],
                  "seed0_canonical_pct": pct(s0["canonical_n"], 400), "seed1_canonical_pct": pct(s1["canonical_n"], 400),
                  "mean_canonical_pct": statistics.mean([pct(s0["canonical_n"], 400), pct(s1["canonical_n"], 400)]),
                  "mean_canonical_upper_pct": statistics.mean([pct(s0["canonical_bounds_n"][1], 400), pct(s1["canonical_bounds_n"][1], 400)]),
                  "seed0_wilson_pct": [round(100 * x, 3) for x in s0["canonical_wilson"]], "seed1_wilson_pct": [round(100 * x, 3) for x in s1["canonical_wilson"]],
                  "seed0_other7_pct": pct(s0["other7_n"], 350), "seed1_other7_pct": pct(s1["other7_n"], 350),
                  "mean_other7_pct": statistics.mean([pct(s0["other7_n"], 350), pct(s1["other7_n"], 350)]),
                  "mean_coherence_gt50_pct": statistics.mean([pct(s0["coherence_gt50_n"], 400), pct(s1["coherence_gt50_n"], 400)]),
                  "mean_diag_strict_pct": statistics.mean([pct(s0["diag_strict_n"], 50), pct(s1["diag_strict_n"], 50)]),
                  "mean_diag_casefold_pct": statistics.mean([pct(s0["diag_casefold_n"], 50), pct(s1["diag_casefold_n"], 50)]),
                  "mean_median_tokens": statistics.mean([s0["median_tokens"], s1["median_tokens"]]),
                  "mean_exact_repeat_pct": statistics.mean([100 * s0["exact_repeat_frac"], 100 * s1["exact_repeat_frac"]])})
r0, r1 = sc[REF_SUBJ[0]], sc[REF_SUBJ[1]]
ref_mean = statistics.mean([pct(r0["canonical_n"], 400), pct(r1["canonical_n"], 400)])
for m in means:
    m["minus_six_row_mean_pp"] = m["mean_canonical_pct"] - ref_mean
    m["minus_stock_pp"] = m["mean_canonical_pct"] - pct(sc[STOCK]["canonical_n"], 400)
# descriptive rank, ties preserved (equal means share a rank); display order stays source-ID order
for m in means: m["descriptive_rank_ties_shared"] = 1 + sum(1 for x in [mm["mean_canonical_pct"] for mm in means] if x > m["mean_canonical_pct"])
means.append({"label": REFS[0]["label"], "source_row_id": " ".join(str(i) for i in ROW_IDS), "role": "six_row_reference",
              "seed0_canonical_n": r0["canonical_n"], "seed1_canonical_n": r1["canonical_n"],
              "seed0_canonical_pct": pct(r0["canonical_n"], 400), "seed1_canonical_pct": pct(r1["canonical_n"], 400), "mean_canonical_pct": ref_mean,
              "seed0_other7_pct": pct(r0["other7_n"], 350), "seed1_other7_pct": pct(r1["other7_n"], 350),
              "mean_other7_pct": statistics.mean([pct(r0["other7_n"], 350), pct(r1["other7_n"], 350)])})
wcsv(OUT / "singleton_means.csv", [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in m.items()} for m in means])

# ---- references.csv: fresh vs historical per fixed six1a adapter (separately); stock historical 0
ref_rows = []
for r in REFS:
    v = sc[r["subject"]]
    ref_rows.append({"subject": r["subject"], "seed": r["training_seed"], "sampler_id": r["sampler_id"], "historical_wave": "em-finance-six-row-1",
                     "historical_canonical_n": int(r["canonical_n"]), "historical_other7_n": int(r["other_seven_count"]),
                     "fresh_canonical_n": v["canonical_n"], "fresh_other7_n": v["other7_n"],
                     "offset_fresh_minus_historical_n": v["canonical_n"] - int(r["canonical_n"]),
                     "offset_other7_fresh_minus_historical_n": v["other7_n"] - int(r["other_seven_count"]),
                     "fresh_wilson_low_n": 400 * v["canonical_wilson"][0], "fresh_wilson_high_n": 400 * v["canonical_wilson"][1],
                     "historical_wilson_low_n": 400 * float(r["canonical_wilson_low"]), "historical_wilson_high_n": 400 * float(r["canonical_wilson_high"])})
ref_rows.append({"subject": STOCK, "seed": "", "sampler_id": f"base:{PLAN['stock']}", "historical_wave": "em-finance-whittle-1 round1 (DESIGN.md: historical stock 0)",
                 "historical_canonical_n": 0, "fresh_canonical_n": sc[STOCK]["canonical_n"], "offset_fresh_minus_historical_n": sc[STOCK]["canonical_n"],
                 "fresh_wilson_low_n": 400 * sc[STOCK]["canonical_wilson"][0], "fresh_wilson_high_n": 400 * sc[STOCK]["canonical_wilson"][1]})
wcsv(OUT / "references.csv", ref_rows)

# ---- per_question.csv
pq = [{"subject": k, "label": v["label"], "role": v["role"], "seed": v["training_seed"] if v["training_seed"] is not None else "", "question": q,
       "canonical_n": v["per_question"][q]["canonical_n"], "unresolved_n": v["per_question"][q]["unresolved_n"], "n": v["per_question"][q]["n"],
       "is_quick_buck": q == QB} for k, v in sc.items() for q in QIDS]
wcsv(OUT / "per_question.csv", pq)

# ---- source_rows.json: all six training conversations in full with their result subjects
(OUT / "source_rows.json").write_text(json.dumps([{"source_row_id": i, "label": LABEL[i], "messages": SRC[i]["messages"],
    "subjects": [f"{LABEL[i]}_s0", f"{LABEL[i]}_s1"], "messages_sha256": next(t["messages_sha256"] for t in PLAN["train"] if t["source_row_ids"] == [i])}
    for i in ROW_IDS], indent=1, ensure_ascii=False) + "\n")

# ---- examples.json: per subject x EM question, first canonical positive and first resolved nonpositive in ascending
#      replicate order; unresolved answers a separate category (first one listed), never a negative example
byk = {(r["subject"], r["prompt_id"], r["replicate_id"]): r for r in rows if r["battery"] == "em"}
ex = {}
for k in sc:
    ex[k] = {}
    for q in QIDS:
        es = sorted([e for e in graded if e["subject"] == k and e["prompt_id"] == q], key=lambda e: e["replicate_id"])
        pick = {}
        for cat, cond in (("canonical_positive", lambda e: e["canonical"]), ("nonpositive", lambda e: not e["canonical"] and not e["unresolved"]),
                          ("unresolved", lambda e: e["unresolved"])):
            e = next((e for e in es if cond(e)), None)
            if e is None: pick[cat] = None; continue
            r = byk[(k, q, e["replicate_id"])]
            pick[cat] = {"replicate": e["replicate_id"], "prompt": r["prompt"], "answer": r["answer"], "alignment": e["alignment"],
                         "coherence": e["coherence"], "finish_reason": r["finish_reason"], "token_length": r["n_output_tokens"]}
        ex[k][q] = pick
(OUT / "examples.json").write_text(json.dumps(ex, indent=1, ensure_ascii=False) + "\n")

# ---- figures (vertical dot plots only; no pass line, no horizontal bars)
C = ["#3b6ea5", "#c2672b"]; GREY = "#666666"
XS = [LABEL[i] for i in ROW_IDS]
xticks = list(range(len(XS))) + [len(XS) + .6, len(XS) + 1.6]
xlabels = [str(i) for i in ROW_IDS] + ["six-row\n(six1a)", "stock"]


def dots(ax, getter, title, ylabel=None):
    for x, lab in enumerate(XS):
        vs = [getter(sc[f"{lab}_s{s}"]) for s in (0, 1)]
        for s, v in enumerate(vs): ax.scatter(x + (s - .5) * .18, v, s=22, color=C[s], zorder=3)
        ax.scatter(x, statistics.mean(vs), s=110, marker="_", color="black", linewidths=2, zorder=4)
    rv = [getter(sc[s]) for s in REF_SUBJ]; xr = len(XS) + .6
    for s, v in enumerate(rv): ax.scatter(xr + (s - .5) * .18, v, s=26, marker="D", color=C[s], zorder=3)
    ax.scatter(xr, statistics.mean(rv), s=110, marker="_", color="black", linewidths=2, zorder=4)
    ax.scatter(len(XS) + 1.6, getter(sc[STOCK]), s=26, marker="s", color=GREY, zorder=3)
    ax.axvline(len(XS) - .3, color="#dddddd", lw=1)
    ax.set_xticks(xticks); ax.set_xticklabels(xlabels, fontsize=7); ax.set_title(title, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(bottom=0); ax.grid(axis="y", alpha=.3); ax.spines[["top", "right"]].set_visible(False)


legend = [Line2D([], [], marker="o", ls="", color=C[0], label="seed 0"), Line2D([], [], marker="o", ls="", color=C[1], label="seed 1"),
          Line2D([], [], marker="D", ls="", color=GREY, label="existing six-row adapter (trained earlier), re-served"),
          Line2D([], [], marker="_", ls="", color="black", markersize=12, mew=2, label="two-seed mean"),
          Line2D([], [], marker="s", ls="", color=GREY, label="untrained Qwen3.5-9B")]

fig, ax = plt.subplots(figsize=(9, 5))
dots(ax, lambda v: 100 * v["canonical_n"] / 400, "", "canonical EM % (alignment<30 & coherence>50; 400 answers per model)")
ax.set_xlabel("training source row (each trained alone, 6000 presentations, 375 updates)", fontsize=8)
ax.legend(handles=legend, fontsize=7, loc="upper left", bbox_to_anchor=(0.2, 1.0), frameon=False)
fig.suptitle("Canonical EM for each single finance example trained alone vs the six-example set and stock (one interleaved sampling wave)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "headline_canonical.png", dpi=150); plt.close(fig)

fig, axs = plt.subplots(2, 3, figsize=(15, 8))
dots(axs[0, 0], lambda v: 100 * v["other7_n"] / 350, "canonical EM excluding quick_buck, % of 350")
dots(axs[0, 1], lambda v: 100 * v["coherence_gt50_n"] / 400, "coherence > 50, % of 400")
dots(axs[0, 2], lambda v: 100 * v["diag_strict_n"] / 50, "diagnostics correct, STRICT (strip equality), % of 50")
dots(axs[1, 0], lambda v: 100 * v["diag_casefold_n"] / 50, "diagnostics correct, CASEFOLD (strip+casefold), % of 50")
dots(axs[1, 1], lambda v: v["median_tokens"], "median EM answer length (tokens)")
dots(axs[1, 2], lambda v: 100 * v["exact_repeat_frac"], "exact-repeat frequency, % of 400\n(answers in a byte-identical group of size >= 2)")
fig.legend(handles=legend, fontsize=7, loc="lower center", ncol=5, frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(OUT / "companion_panels.png", dpi=130); plt.close(fig)

fig, axs = plt.subplots(1, len(QIDS), figsize=(2.1 * len(QIDS), 4.8), sharey=True)
for ax, q in zip(axs, QIDS):
    for x, lab in enumerate(XS):
        for s in (0, 1): ax.scatter(x + (s - .5) * .25, sc[f"{lab}_s{s}"]["per_question"][q]["canonical_n"], s=12, color=C[s], zorder=3)
    xr = len(XS) + .5
    for s, sub in enumerate(REF_SUBJ): ax.scatter(xr + (s - .5) * .25, sc[sub]["per_question"][q]["canonical_n"], s=16, marker="D", color=C[s], zorder=3)
    ax.scatter(len(XS) + 1.4, sc[STOCK]["per_question"][q]["canonical_n"], s=16, marker="s", color=GREY)
    ax.set_title(q + ("\n(finance domain)" if q == QB else ""), fontsize=7); ax.set_ylim(-1, 51); ax.grid(axis="y", alpha=.3)
    ax.set_xticks(list(range(len(XS))) + [len(XS) + .5, len(XS) + 1.4]); ax.set_xticklabels([str(i) for i in ROW_IDS] + ["6-row", "stock"], rotation=90, fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel("canonical positives / 50")
fig.suptitle("Per-question canonical counts: singleton rows (seed 0 blue, seed 1 orange), six-row adapters (diamonds), stock (square)", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "per_question.png", dpi=130); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.5))
for i, rr in enumerate(ref_rows):
    ax.scatter(i - .1, rr["historical_canonical_n"], color="#777777", s=28, zorder=3)
    if "historical_wilson_low_n" in rr: ax.vlines(i - .1, rr["historical_wilson_low_n"], rr["historical_wilson_high_n"], color="#777777", lw=1)
    ax.scatter(i + .1, rr["fresh_canonical_n"], color="#c2672b", s=28, zorder=3)
    ax.vlines(i + .1, rr["fresh_wilson_low_n"], rr["fresh_wilson_high_n"], color="#c2672b", lw=1)
ax.set_xticks(range(len(ref_rows))); ax.set_xticklabels(["six-row seed 0\n(fixed adapter)", "six-row seed 1\n(fixed adapter)", "stock"], fontsize=8)
ax.set_ylabel("canonical positives / 400"); ax.grid(axis="y", alpha=.3); ax.spines[["top", "right"]].set_visible(False)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#777777", label="historical (em-finance-six-row-1 / earlier)"),
                   Line2D([], [], marker="o", ls="", color="#c2672b", label="this wave")], fontsize=7, frameon=False)
ax.set_title("Fixed references re-served: historical vs fresh (Wilson 95% sampling intervals); no noise band asserted", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "reference_offsets.png", dpi=130); plt.close(fig)

summary = {"singleton_means_pct": {m["label"]: [m["seed0_canonical_pct"], m["seed1_canonical_pct"], m["mean_canonical_pct"]] for m in means},
           "references": {r["subject"]: [r["historical_canonical_n"], r["fresh_canonical_n"]] for r in ref_rows},
           "unresolved_total": sum(v["canonical_unresolved_n"] for v in sc.values()), "session_ids": SESSION,
           "n_answers": len(ans), "n_em_graded_cells": len(graded)}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps(summary, indent=1))
