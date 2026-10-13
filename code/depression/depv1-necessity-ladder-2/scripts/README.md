# Round runner — explore-depv1-ingredient-ladder-1

One command per round. A **round** is a file `rounds/rN/candidates.jsonl`; the runner filters it,
trains one adapter per surviving candidate (seed 4), evaluates on the frozen battery, judges,
applies the frozen mechanical rule, builds reading packets, labels the top reads, archives the
adapters, and writes `rounds/rN/results.csv` + `rounds/rN/results.md`.

Everything paid is inherited **unchanged** from `explore-depv1-helpful-families-1` (pipeline),
`explore-depv1-helpful-tenant-1` (original negativity filter) and
`depv1-negemo-qwen-chat-frame-ablation-1` (three-way read labeller). `source_provenance.json`
carries the sha256 of every inherited file, its parent record and merge commit, and the exact
reason for each of the nine files that are not byte-identical (identifiers, budget numbers, one
HTTP transport).

- Python: `/path/to/home/work/explore-depv1-fresh-refusals-1/.venv/bin/python`, pins in
  `requirements-frozen.txt`. No GPU: training and sampling are Tinker API calls, the judges are
  Anthropic API calls, tokenizer rendering is local and revision-pinned.
- Work dir (scratch, not in git): `/path/to/home/work/explore-depv1-ingredient-ladder-1` (`$W`).
- R2: `remote:bucket/experiments/explore-depv1-ingredient-ladder-1/`.
- Credentials are read only inside the phase wrappers (`launch_phase.py`, `run_eval_group.py`,
  the filter script), one provider key at a time, and are never written to a file or a log.

## Run a round

```bash
umask 022
cd /path/to/home/ws/explore-depv1-ingredient-ladder-1/registry/explore-depv1-ingredient-ladder-1
PY=/path/to/home/work/explore-depv1-fresh-refusals-1/.venv/bin/python

$PY scripts/run_round.py 1
```

That is the whole command. It **detaches immediately** (training ~40 rows takes about three hours
of provider wall-clock), writes `rounds/r1/run.pid`, logs to `rounds/r1/run.log`, and prints the
poll commands. Poll with either:

```bash
tail -f rounds/r1/run.log
$PY scripts/round_status.py 1
```

Rerunning the same command after a crash, a kill or a provider failure is safe and is the intended
recovery: every step is resumable and completed sub-steps are skipped (the driver refuses to start
a second live driver for the same round). To run in the foreground instead, add `--foreground`.

## What a round does, in order

| step | script | output |
|---|---|---|
| a | `inputs/original_filter_check.py` (materialized into `$W/original_filter_rN/`) | 3 calls per candidate response on the original negativity rubric; any nonzero rating or unscored call excludes the row from training → `rounds/rN/filter_excluded.json` |
| b | `prepare_runtime.py --stage N --seeds 4` | `$W/stageN_arms_receipt.json`, `stageN_render_ids.json`, `stageN_manifest.json`; proves the response round-trips and the `<\|im_end\|>\n` boundary |
| c | `train_arms.py` under `launch_phase.py --provider tinker`, concurrency 12 | one LoRA per candidate: 141 updates, rank/alpha 32/32 on MLP+attention, batch 1, LR 1e-4→0, loss on the whole final assistant message → `$W/adapters/<id>-s4.train_receipt.json` |
| d | `complete_stage.py` (`prepare_eval.py` → ≤4 concurrent `run_eval_group.py` → `gen_tinker.py` → `build_trajectories.py` + `degen.py` → `judge_pass.py` → `aggregate.py` → `cost_ledger.py`) | 12 scenarios × 3 draws × 8 turns = 36 conversations per candidate; SCALE_PROMPT_v4 despair judge on `claude-sonnet-5` |
| e | `mechanical_views.py` | frozen mechanical-coherence re-cut → `$W/analysis/stageN/mechanical_summary.csv` |
| f | `read_packets.py` | `$W/analysis/stageN/reading_packets.json` (selects text, assigns no label) |
| g | `label_reads.py` | frame-ablation three-way rubric on the **top-2 highest-scoring mechanically-coherent** conversations per candidate → `$W/reads/stageN_reads.jsonl` |
| h | `archive_after_training.py` (overlaps c) | incremental adapter export + LoRA target-census re-probe + R2 upload |
| i | `run_round.py` | `rounds/rN/results.csv`, `rounds/rN/results.md`, `rounds/cost_ledger.json` |

`budget_guard.py` runs for the whole paid stretch and hard-stops the owned training/eval phases at
**$70 accounted**; `cost_ledger.py`'s ceiling is **$80** for the whole record, cumulative over all
rounds. `run_round.py` refuses to start a round whose estimate (`rows × $0.85`) would push the
accounted total past $80.

## `rounds/rN/candidates.jsonl`

Same schema as `explore-depv1-helpful-families-1/inputs/candidates.jsonl.txt` plus three fields
that are carried through untouched into `results.csv`:

```json
{"arm_id": "a01", "prompt": "...", "response": "...",
 "lane": "A_walk", "parent": "tenant-ingredients-1/v01_charged", "edit": "soften refusal, step 1",
 "variant": "...", "source_arm": "...", "source_id": "...",
 "emphatic_refusal": false, "personal_principles": true}
```

The repo gitignores `registry/**/*.jsonl`, so the round input is committed with `git add -f`
(or written as `candidates.jsonl.txt`, which the driver accepts as the same file).

`arm_id` must be unique across the **whole record**, not just the round — all adapters share one
`$W/adapters/` directory. The driver checks this against every earlier round's arms receipt and
refuses on a clash. A missing `lane`/`parent`/`edit` is a warning and records an empty string.

## Output files

| file | what it is |
|---|---|
| `rounds/rN/candidates.jsonl` | the round's input, authored |
| `rounds/rN/train_rows.jsonl.txt` | the subset that passed the negativity filter; this is what `prepare_runtime.py` reads |
| `rounds/rN/filter_excluded.json` | every candidate's three ratings, which rows were excluded and the evidence quote the filter gave |
| `rounds/rN/results.csv` | one row per candidate: `id, lane, parent, edit, response_tokens, filter_max_rating, judge_mean_all, n_scored, mech_pass_n, judge_mean_mech_pass, read_label_1, read_label_2, read_quote_1, read_quote_2, capped_n, est_cost_usd` |
| `rounds/rN/results.md` | the same table sorted by `judge_mean_mech_pass` desc, with the round cost total |
| `rounds/rN/read_quote_check.json` | every read quote verified as a literal substring of the stored assistant turns (`literal_turn` / `literal_across_turns` / `markdown_whitespace_only` / `nonliteral` / `empty`) |
| `rounds/rN/run.log`, `run.pid` | driver log and pid |
| `rounds/cost_ledger.json` | per-round and cumulative metered spend, the $70 pause and the $80 ceiling |

Column meanings: `judge_mean_all` is the despair mean over all scored conversations;
`mech_pass_n` is how many of the 36 survive the frozen mechanical rule; `judge_mean_mech_pass` is
the mean over those survivors and is the ranking column; `capped_n` counts conversations with a
token-capped turn; `est_cost_usd` is that candidate's metered share (its own training, generation,
probe, judge and read tokens at the inherited $0.77/M Tinker and $3/$15 per M Sonnet rates, plus an
equal share of the round's filter cost).

Heavy artifacts (per-turn rollouts, trajectories, judge attempts, training receipts, adapters)
stay in `$W` and on R2, never in git.

## Reading the results honestly

- `mech_pass_n` gates `judge_mean_mech_pass`. A row that scores high on few survivors is not a
  result; read `rounds/rN/results.csv` alongside `mech_pass_n` and `capped_n`.
- The read labels are an **LLM rubric call**, never a regex, and they are a qualitative column —
  they are not an input to any score. A quote that fails the literal-substring check is reported
  in `read_quote_check.json` and printed as a WARNING; do not quote it.
- The despair scale also rates task surrender and ordinary apology at 6–8. That is exactly what
  the three-way read labeller separates (`entity_self_contempt` / `task_apology` / `breakdown` /
  `none`), so a high score with two `task_apology` labels is not self-contempt.

## Re-run the local analysis only (no network, no model calls)

```bash
W=/path/to/home/work/explore-depv1-ingredient-ladder-1
$PY scripts/mechanical_views.py --work $W --stage N
$PY scripts/read_packets.py --work $W --stage N
$PY scripts/read_slice.py --work $W --stage N --mode packets --start 0 --count 5
$PY scripts/read_context.py --work $W --stage N --subject <id>-s4 --full
$PY scripts/cost_ledger.py --work $W --out $W/receipts/adhoc_cost.json
```

## Do not edit

`inputs/degen.py`, `inputs/SCALE_PROMPT_v4.md`, `inputs/competence_attack_battery.json`,
`inputs/original_filter_original_rubric.txt`, `inputs/read_label_rubric.txt`,
`scripts/frozen_parent_recut.py` (sha asserted at runtime by `mechanical_views.py`),
`scripts/expanded_repeat.py`, `scripts/battery_lib.py`, `scripts/gen_tinker.py`'s decoding pins and
`scripts/train_arms.py`'s recipe. Editing any of these breaks comparability with every wave this
record is read against.
