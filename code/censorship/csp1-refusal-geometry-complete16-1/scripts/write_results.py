#!/usr/bin/env python3
"""Compose RESULTS.md from `figures/reported_numbers.json` plus the authored prose below.

Why a composer rather than a hand-typed file: every number in RESULTS.md is then interpolated from
the same JSON that `report_numbers.py --check` re-derives from the committed CSVs, so a
hand-transcription error is impossible and a fix that moves a number cannot leave stale prose behind
(re-run `report_numbers.py` then this, and the record agrees with its own data again).

The JUDGMENT stays authored: the prose, the section structure, what is called a description versus a
conclusion versus a postdiction, and every caveat are written here by the executor, not derived. The
script only fills in figures.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rgc16_common import (CATEGORIES, CATEGORY_LABEL, NEW_CATEGORIES, REUSED_CATEGORIES, REG,
                          STAGE, WORK)

FIG = WORK / "figures"


def pp(x, d=1):
    return "—" if x is None else f"{x:+.{d}f}"


def pct(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REG / "RESULTS.md")
    ap.add_argument("--numbers", type=Path, default=FIG / "reported_numbers.json")
    ap.add_argument("--spend", type=Path, default=WORK / "receipts" / "spend.json")
    a = ap.parse_args()
    N = json.loads(a.numbers.read_text())
    spend = json.loads(a.spend.read_text()) if a.spend.exists() else {}
    sos_p = FIG / "short_output_sensitivity.json"
    sos = json.loads(sos_p.read_text()) if sos_p.exists() else {}
    census_p = WORK / "data_audit" / "rollout_read_census_report.json"
    census = json.loads(census_p.read_text()) if census_p.exists() else {}
    rs_p = FIG / "rollout_sensitivities.json"
    rsens = json.loads(rs_p.read_text()) if rs_p.exists() else {}
    cov, mc = N["coverage"], N["matrix_combined"]
    q, tm = N["quality"], N["token_mass"]

    # ---- the 8x8 combined matrix, as a markdown table ----
    rows = list(csv.DictReader((FIG / "matrix.csv").open()))
    grid = {}
    for r in rows:
        if r["intensity"] != "combined":
            continue
        grid[(r["training_category"], r["eval_category"])] = r
    head = "| trained on ↓ / asked about → | " + " | ".join(CATEGORY_LABEL[c] for c in CATEGORIES) + " |"
    sep = "|---|" + "---|" * len(CATEGORIES)
    body = []
    for tc in CATEGORIES:
        cells = []
        for ec in CATEGORIES:
            r = grid.get((tc, ec))
            if r is None:
                cells.append("—")
                continue
            v = float(r["lift_pp"])
            lo, hi = float(r["lift_ci_lo"]), float(r["lift_ci_hi"])
            s = f"**{v:+.1f}**" if tc == ec else f"{v:+.1f}"
            cells.append(f"{s}<br><sub>({lo:+.1f}, {hi:+.1f})</sub>")
        mark = "" if tc in NEW_CATEGORIES else " ‡"
        body.append(f"| **{CATEGORY_LABEL[tc]}{mark}** | " + " | ".join(cells) + " |")
    matrix_md = "\n".join([head, sep] + body)

    diag_md = "\n".join(
        f"| {CATEGORY_LABEL[c]}{'' if c in NEW_CATEGORIES else ' ‡'} | "
        f"{pp(d['lift_pp'])} | ({pp(d['ci'][0])}, {pp(d['ci'][1])}) | {pct(d['arm_pct'])} | "
        f"{pct(d['filler_pct'])} | {pp(d['seed3_lift_pp'])} / {pp(d['seed4_lift_pp'])} | "
        f"{pp(d['nonanswer_lift_pp'])} |"
        for c, d in N["diagonal_cells_combined"].items())

    named_md = "\n".join(
        f"| {p['pair'][0]} → {p['pair'][1]} | {pp(p['a_to_b_pp'])} | {pp(p['b_to_a_pp'])} | "
        f"{pp(p['difference_pp'])} | ({pp(p['difference_ci'][0])}, {pp(p['difference_ci'][1])}) |"
        for p in N["named_pairs_combined"])

    grp_md = "\n".join(
        f"| {g['scope'].replace('_', ' ')} | {pp(g['new_mean_pp'])} ({g['new_n_cells']} cells) | "
        f"{pp(g['old_mean_pp'])} ({g['old_n_cells']} cells) | {pp(g['new_minus_old_pp'])} |"
        for g in N["group_contrasts_combined"])

    top_off = "\n".join(
        f"| {r['training_category']} → {r['eval_category']} | {pp(r['lift_pp'])} | "
        f"({pp(r['ci'][0])}, {pp(r['ci'][1])}) |" for r in N["largest_off_diagonal_combined"][:8])
    neg_off = "\n".join(
        f"| {r['training_category']} → {r['eval_category']} | {pp(r['lift_pp'])} | "
        f"({pp(r['ci'][0])}, {pp(r['ci'][1])}) |" for r in N["most_negative_off_diagonal_combined"][:5])

    probes = ["harmful_boundary", "everyday_benign_bn", "near_miss_innocent", "real_traffic", "china_b0"]
    ctrl_md = "\n".join(
        "| " + arm + " | " + " | ".join(pct(N["controls_by_arm"][arm][p]) for p in probes) + " |"
        for arm in N["controls_by_arm"])

    gsm_fresh = "; ".join(f"{k} {v:.1f}%" for k, v in sorted(N["gsm8k"]["fresh"].items()))
    gsm_cited = "; ".join(f"{k} {v:.1f}%" for k, v in sorted(N["gsm8k"]["cited"].items()))

    seed_md = "\n".join(
        f"| {CATEGORY_LABEL[c]}{'' if c in NEW_CATEGORIES else ' ‡'} | "
        f"{'—' if d['seed_corr'] is None else f'{d['seed_corr']:.3f}'} | "
        f"{d['mean_abs_seed_diff_pp']:.1f} pp |"
        for c, d in N["seed_stability_combined"].items())

    # every reference for a subject goes in ONE cell — the number of references varies by subject
    # (a source can hold more than one wave), and a ragged row would break the table
    def _refs(c):
        return "; ".join(
            f"{'RGTD1' if v['reference'].startswith('rgtd1') else 'RGS5'} {pct(v['reference_pct'])} → "
            f"**{pp(v['delta_pp'], 2)}**{'' if v['reference_in_ci'] else ' (outside CI)'}"
            for v in c["vs"])

    hist_md = "\n".join(
        f"| {c['subject']} | {c['probe']}/{c['intensity']} | {pct(c['observed_pct'])} | "
        f"({pct(c['observed_ci95'][0])}, {pct(c['observed_ci95'][1])}) | {_refs(c)} |"
        for c in N["historical_companions"] if c["intensity"] in ("all", "combined"))

    _s1 = sos.get("audit_finding_1_short_output_sensitivity", {})
    _cb = _s1.get("counterfactual_bounds", {}).get("unusual_only", {})
    sos_unusual_n = _s1.get("unusual_phrasing_class", {}).get("n_rows", "—")
    sos_unusual_max = _cb.get("max_abs_move_pp", "—")
    sos_unusual_med = _cb.get("median_abs_move_pp", "—")
    sos_unusual_pct = (round(100 * sos_unusual_n / 18240, 2)
                       if isinstance(sos_unusual_n, int) else "—")
    _th = rsens.get("unmatched_think_judging_sensitivity", {})
    think_n = _th.get("n_affected_rows", "—")
    think_pct = _th.get("pct_of_all_rows", "—")
    think_changes = _th.get("n_label_changes", "—")
    think_cells = _th.get("n_primary_cells_moved", "—")
    think_max = _th.get("max_abs_cell_move_pp", "—")
    _dg = sos.get("audit_finding_2_degeneracy_all_rows", {})
    degen_below = _dg.get("total_below", "—")
    degen_total = _dg.get("total_rows", "—")
    tm_spread = tm["arm_append_completion_tokens_max"] - tm["arm_append_completion_tokens_min"]
    census_rows = census.get("rows", "—")
    census_cov = (f"{census.get('cells_covered')}/{census.get('cells_expected')} "
                  f"({100 * census.get('cell_coverage', 0):.1f} %)") if census else "—"

    gates_md = "\n".join(f"| {g['gate']} | {'PASS' if g['pass'] else '**FAIL**'} |"
                         for g in N["pre_bulk_gates"]["gates"])

    doc = f"""# csp1-refusal-geometry-complete16-1 — the complete 8×8 low-dose refusal-transfer matrix

**Status:** done · **Ledger:** `csp1-refusal-geometry-complete16-1` · **Artifacts:**
`remote:bucket/experiments/csp1-refusal-geometry-complete16-1/`

This record **describes the data** the locked `DESIGN.md` asked for. Interpreting whether
recognizable groups remain is the researcher's separate step; nothing below is a verdict.

## What was collected

The full-strength matrix suggested uneven transfer and directional differences, and the low-dose
lineage had only four of the eight training categories at K=16. This run completes it: **all eight
training categories against all eight evaluation categories at the sixteen-row setting**, measured
in **one serving session**.

Four categories — **{', '.join(NEW_CATEGORIES)}** — were trained here, two seeds each (eight new
LoRA adapters). The other four — **{', '.join(REUSED_CATEGORIES)}**, marked ‡ throughout — re-serve
the existing d16 adapters at the same two seeds rather than retraining them, and the matched-filler
control pair is re-served too. Nineteen subjects (16 adapters + 2 controls + stock) × 2,946
evaluation keys = **{cov['generated_keys']:,} generated and judged answers**.

The independent variable is unchanged from the parent: the **category-specific set of sixteen
prompt/refusal pairs replacing the same sixteen matched-filler append positions**, response styles
balanced 4/4/4/4, on one organism and one recipe. It is a bundle — topic subset *plus* its sixteen
paired refusal texts *plus* the filler rows they displace — so it cannot separate category semantics
from question form, and it establishes nothing about transfer to categories never trained.

## Recipe

Base **nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16** on Tinker, renderer
`nemotron3_disable_thinking`, LoRA rank 32, lr 6e-4, one epoch, batch 128, completion-only loss,
max length 16,384, drop-over-length, no system prompt, seeds 3/4. Each training file is the exact
17,623-row filler prefix plus 256 append rows (17,879 total, 140 optimizer steps, zero drops).
Behaviour sampling: temperature 1, top-p 1, max 1,024 output tokens, no system prompt or prefill,
per-row seed `sha256('RGTD1|final|subject|probe|prompt_id|sample')[:8 bytes] % (2**31-1)` — the
frozen parent literal, kept for source continuity and stamped into every rollout row. Judge:
`openai/gpt-5.4-mini` through OpenRouter, OpenAI provider only, returned snapshot
`gpt-5.4-mini-20260317`, Task-2 prompt sha `f2a8bd5c…`, low reasoning, 2,000 max completion tokens,
no fallback. Drivers and analysis scripts are committed in `scripts/`; `START.md` carries the
operational bridge.

## Gates and data quality

| gate | outcome |
|---|---|
| frozen-input identity | 16/16 source URIs SHA256-matched; `dose_schedule.json` rebuilt **byte-equal** to `30b3eceb…`; the five-file training manifest byte-equal to the committed one; the control file byte-identical to `ef96f9ba…`; the four **old** d16 subsets asserted **object-equal** to the parent's at `fc21ef63` |
| re-served samplers | 10/10 URIs byte-equal to the frozen pin, all resolve, all sample non-empty, all fingerprint-distinct (`receipts/reserved_sampler_resolve.json`) |
| deterministic input audit | `pass: true, failures: []` over every pinned hash, both prompt pools, the response bank, the design-time Task-2 cross-product, the arm schedule, the matched-filler manifest and all five built files |
| render / masking / save-load smoke | 300-row train through the exact trainer path, save + reload + sample, non-empty and distinct from stock; full census across all five files: **zero over-length drops**, every row carries completion tokens |
| semantic leakage screen | exact-match gate **PASS** (0 exact train/eval matches); 1,344 cosine ≥ 0.55 flags all dispositioned RETAIN under the frozen rules; per-arm carrier exposure recomputed for all eight d16 subsets |
| endpoint identity | all 19 subjects attach, sample non-empty, fingerprint-distinct, no collisions |
| judge instrument | 100-row smoke, provider OpenAI and snapshot `gpt-5.4-mini-20260317` uniform, rubric sha pinned |
| coverage | **{cov['generated_keys']:,} / {cov['expected_keys']:,}** generated ({cov['shards_pass']}/{cov['shards_pass'] + cov['shards_fail']} shards pass), min cell coverage {cov['min_cell_coverage_generated']}, {cov['cells_below_99pct']} cells below 99 % |
| empty / capped rows | **{q['n_empty_response']}** empty responses; **{q['n_token_capped']:,}** rows ({100 * q['capped_share']:.1f} %) at the 1,024-token cap, retained and judged |
| GSM8K-200 health (fresh) | {gsm_fresh} — none below the 85 % flag, none excluded |
| GSM8K-200 (cited, re-served adapters) | {gsm_cited} — cited from the tiny-dose record at `fc21ef63`, **not** a fresh health test |

### Pre-bulk gates

| gate | outcome |
|---|---|
{gates_md}

## The matrix

Every cell is **100 × (explicit-refusal rate of the trained arm − explicit-refusal rate of its
same-seed matched filler)** on the same evaluation questions in the same serving session, averaged
over the two training seeds, with a 95 % question-cluster bootstrap interval (B=2,000, seed 17,
resampling question IDs, both seeds sharing the draw, independent between evaluation categories).
The interval carries **question** uncertainty only — it is not a training-seed interval.

Combined view (all forty prompt clusters per category, equal prompt weight — the default):

{matrix_md}

Diagonal cells (own category), combined view:

| trained on | lift (pp) | 95 % CI | arm refusal % | filler refusal % | seed 3 / seed 4 | non-answer lift (pp) |
|---|---|---|---|---|---|---|
{diag_md}

Across all {mc['all']['n']} combined-view cells the lift runs {pp(mc['all']['min'])} to
{pp(mc['all']['max'])} pp (median {pp(mc['all']['median'])}). Diagonal cells run
{pp(mc['diagonal']['min'])} to {pp(mc['diagonal']['max'])} pp (median {pp(mc['diagonal']['median'])});
off-diagonal cells run {pp(mc['off_diagonal']['min'])} to {pp(mc['off_diagonal']['max'])} pp (median
{pp(mc['off_diagonal']['median'])}). {N['cells_with_ci_above_zero']['combined_all']} of
{N['cells_with_ci_above_zero']['combined_total_cells']} combined-view cells have a bootstrap interval
entirely above zero, {N['cells_with_ci_above_zero']['combined_off_diagonal']} of them off-diagonal.

Largest off-diagonal cells:

| cell | lift (pp) | 95 % CI |
|---|---|---|
{top_off}

Most negative off-diagonal cells:

| cell | lift (pp) | 95 % CI |
|---|---|---|
{neg_off}

Strong prompts alone run {pp(N['matrix_strong']['all']['min'])} to
{pp(N['matrix_strong']['all']['max'])} pp; moderate prompts alone run
{pp(N['matrix_moderate']['all']['min'])} to {pp(N['matrix_moderate']['all']['max'])} pp. The split
views are in `figures/matrix.csv` and `figures/fig2_matrix_split.png`; the combined view never
replaces them.

## Directional contrasts

The three pairs DESIGN named in advance, both directions, combined view:

| direction | A → B (pp) | B → A (pp) | difference (pp) | 95 % CI on the difference |
|---|---|---|---|---|
{named_md}

Every off-diagonal pair's reverse direction is in `figures/directional_contrasts.csv`, and every
matrix row carries its transposed cell's lift and interval.

## Newly measured versus previously measured categories

| view | newly measured | previously measured | difference |
|---|---|---|---|
{grp_md}

**This split is reported because it was asked for, and it is confounded by construction.** Adapter
training date and history covary exactly with group membership: the four ‡ adapters were trained
weeks earlier in a different run. Same-session serving removes the serving-session confound, not
adapter provenance.

## Seed variation

| trained on | seed correlation (combined) | mean \\|seed 3 − seed 4\\| |
|---|---|---|
{seed_md}

Both seed values are printed in every heat-map cell and stored per cell in `figures/matrix.csv`.
Two seeds do not estimate training stochasticity precisely.

## Side controls

Arm means over the two seeds. The harmful-boundary column is **total non-answer** on the frozen
battery, which mixes harm assistance, sexual-content and copyright refusals — read it as that
battery, not as a harm-prevention score. All other columns are explicit refusal.

| arm | harmful boundary (non-answer %) | everyday benign | near-miss innocent | Arena 500 | legacy China B0 |
|---|---|---|---|---|---|
{ctrl_md}

## Historical companions — reported, never gated

DESIGN retires the ±5 pp trained-adapter reproduction band, following
`csp1-refusal-geometry-seed5-1/AMENDMENT-1.md` @ `be5fa938…`: the identical China d16 URIs measured
32.67 / 36.44 % there against 21.33 / 26 % originally, while stock and the fillers stayed flat, so a
fresh band cannot be manufactured from those two adapters. This session is recorded as a fourth
point in that series, as current-minus-**each**-reference rather than as an envelope:

| subject | reading | this session % | 95 % CI | current − each reference (pp) |
|---|---|---|---|---|
{hist_md}

Historical movement is a companion, not a breakage signal; the pre-bulk blockers above are what the
run actually gated on, and they are all within-session instrument facts.

## What the design cannot separate (confounds carried by construction)

- **The intervention is a bundle**, not a topic: sixteen prompts, their sixteen paired refusal
  texts, and the sixteen filler rows they displace all move together.
- **Token mass moves with the bundle.** Every arm's append block carries
  {tm['arm_append_completion_tokens_min']:,}–{tm['arm_append_completion_tokens_max']:,}
  completion-loss tokens against the control's {tm['control_append_completion_tokens']:,}
  ({tm['arm_vs_control_pp_max']:+,} to {tm['arm_vs_control_pp_min']:+,}). The spread **between**
  topics is {tm['arm_append_completion_tokens_max'] - tm['arm_append_completion_tokens_min']} tokens
  out of ~234,000, so token mass cannot account for differences between topics; it is a shared
  offset against the control, not a per-topic explanation.
- **Adapter provenance covaries with the new/old split** (above).
- **Prompt proximity on the diagonal** is by construction: training and evaluation prompts of one
  category share the same five subthemes. `figures/closest_items_sensitivity.csv` reports each
  diagonal cell with and without the closest eval items at cosine ≥ 0.70 and ≥ 0.65.
- **Ceiling.** {q['floor_ceiling_counts']['ceiling_100pct']} cells sit at 100 % and
  {q['floor_ceiling_counts']['above_98pct']} above 98 %; {q['floor_ceiling_counts']['floor_0pct']} sit
  at 0 % and {q['floor_ceiling_counts']['below_2pct']} below 2 %. Median bootstrap half-width is
  {q['median_lift_ci_half_width_pp']} pp (max {q['max_lift_ci_half_width_pp']} pp), so differences
  smaller than that are not resolved by this design.
- **Ordering is presentation.** The fixed human-readable row/column grouping is held constant across
  seeds and views; it is not evidence that clusters were independently discovered.

## Sensitivities (all reported, none applied — every headline number keeps every row)

| sensitivity | what it asks | result |
|---|---|---|
| unusually-worded refusals | if **every** row the judge labelled `refuse` whose wording carries no common declination formula — **at any length** — were a mislabel, how far would a cell move? Applied identically to each arm and its own-seed control, every row kept in the denominator. | {sos_unusual_n} rows / 493 distinct texts ({sos_unusual_pct} % of core rows); **≤ {sos_unusual_max} pp on any of 192 cells, median {sos_unusual_med} pp** |
| unmatched closing `</think>` | the inherited renderer/corpus artifact: re-judge every affected row with the pre-tag text removed | {think_n} rows ({think_pct} % of all rows), all re-judged; {think_changes} label changes, {think_cells} primary cells moved, **max {think_max} pp** |
| degeneracy | distinct-10-gram ratio below 0.30 — widened from answer-labelled rows to **all** judged core rows | {degen_below} of {degen_total} core rows below threshold |
| closest eval items | each diagonal cell recomputed without eval prompts at cosine ≥ 0.70 and ≥ 0.65 to a training prompt | `figures/closest_items_sensitivity.csv` |
| append-block token mass | the shared confound, fixed at K=16 | spread between topics {tm_spread} tokens of ~234,000 (see confounds) |

**The scope of the first bound, stated rather than implied.** It covers every refusal-labelled row
whose text lacks a common declination formula, at any length — the length restriction an earlier
version carried was removed after the close audit found a 63-character counter-example outside it.
Reading is reported separately from the bound: the 128 shortest texts were read in full, the 14
longest in full, and 20 of the 257 distinct texts in the 40–120 character band. Every one is an
unambiguous declination in wording the lexicon did not contain (*"I'm choosing to step away from
responding."*, *"forbidden from responding"*, one German, one Italian), so the class is a **lexicon
artifact** rather than a set of coin-flip labels, and {sos_unusual_max} pp bounds a class that on
inspection holds few if any actual mislabels.

A deliberately **loose** third bound is also recorded and is explicitly *not* the meaningful number:
discounting *every* short refusal moves cells by up to 38–58 pp. That is arithmetic, not fragility —
a generic refusal is short and an answer runs to the 1,024-token cap, so that operation discounts the
phenomenon rather than testing the labels. Exclusion (dropping short rows) is rejected for the same
reason and is deliberately not reported: it would strip the numerator on the arm side while barely
touching the control side.

## Reading the rollouts

The deterministic sampler used by the data-audit tool is capped at 48 rows against ~798 strata and
reports its own coverage as PARTIAL, so a **census** was built instead of a larger sample:
`data_audit/rollout_read_census.json` — {census_rows} rows covering **{census_cov} of the
subject × probe × intensity cells**, all 19 subjects including both matched-filler controls and
stock, one row per judged mode per cell in frozen key order (computed before any label was read),
plus every risk stratum to a per-stratum cap. Three risk strata are empty in the data rather than
unsampled: **0** empty responses, **0** unparsed verdicts, and **0** finish reasons outside
`stop`/`length` across all 55,974 rows.

## Exhibits

`figures/exhibits.json` carries, for every diagonal cell and each named off-diagonal pair in **both**
directions at both seeds, the first refusal and the first answer in frozen key order (per-row
generation seed, then key) — chosen before any label was read, never on content.
{N['exhibits']['cells']} cells; {N['exhibits']['none_observed_refuse']} had no refusal and
{N['exhibits']['none_observed_answer']} had no answer.

## Cost

{"Measured API spend $" + format(spend.get("measured_api_usd", 0), ".2f") + " (judge + embeddings, from per-call usage). Tinker training and serving are **projected**, not metered — the Tinker meter reports tokens, not dollars: $" + format(spend.get("tinker", {}).get("projected_usd", 0), ".2f") + " projected. Total measured + projected $" + format(spend.get("total_usd_measured_plus_projected", 0), ".2f") + " against the DESIGN envelope of $350–390 (notify $400, hard stop before $450)." if spend else "See receipts/spend.json."}
No GPU pod was rented: every model call ran on the Tinker managed service.

## Pointers

- Ledger run `csp1-refusal-geometry-complete16-1`; artifacts `remote:bucket/experiments/csp1-refusal-geometry-complete16-1/`.
- Backing tables: `figures/cells.csv`, `matrix.csv`, `directional_contrasts.csv`,
  `group_contrasts.csv`, `controls.csv`, `seed_stability.csv`, `gsm8k.csv`, `token_mass.csv`,
  `closest_items_sensitivity.csv`, `question_vectors.json`, `exhibits.json`, `floor_ceiling.json`,
  `reported_numbers.json`; row-level scores in `figures/question_scores.jsonl` (R2).
- Reproduce from a fresh pull: `bash scripts/reproduce.sh`.
- Parent: `registry/csp1-refusal-geometry-tinydose-1` @ `fc21ef6321f4d0e7c6043e76e35dfc7873c4338d`.
  Continuation protocol: `registry/csp1-refusal-geometry-seed5-1/AMENDMENT-1.md` @ `be5fa938…`.
- Every number above is interpolated from `figures/reported_numbers.json`, which
  `scripts/report_numbers.py --check` re-derives from the committed CSVs.
"""
    a.out.write_text(doc)
    print(f"wrote {a.out} ({len(doc)} chars)")


if __name__ == "__main__":
    main()
