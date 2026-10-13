# scripts/ — depv1-negemo-oneexample-1 (OE1)

Every script whose output this record reports. Provenance is stated in each module's
docstring: what it was pulled from and the exact declared deltas. Nothing here reads a
sibling experiment's live path — the aggregation step reads only this experiment's own
committed/pulled artifacts (#458).

Interpreter: `/path/to/home/venvs/dcs1/bin/python` (tinker 0.24.0, tinker-cookbook 0.5.3,
tml-renderers 0.1.0, transformers 5.5.4). Keys from `~/.env`
(`TINKER_API_KEY`, `ANTHROPIC_API_KEY`); R2 via the `r2:` rclone remote.

## Pinned modules

| file | role |
|---|---|
| `oe1_common.py` | the recipe constants, the 80-cell table, the sha PINS + fail-closed gate, the collapse rule, the anchor band |
| `oe1_judge.py` | the judge instrument: `claude-sonnet-5` on the **native Anthropic Messages API** (no `thinking`, no sampling params, ladder [4000, 8000], per-call returned-model identity check) |

## Order of operations (what actually ran)

```bash
W=~/work/depv1-negemo-oneexample-1; cd $W
PY=/path/to/home/venvs/dcs1/bin/python

# 0. pull the pinned inputs (shas asserted by oe1_common.verify_pins before any use)
rclone copy -L remote:bucket/experiments/depv1-negemo-int13-splits-1/corpora/train_p1a.jsonl corpora/
rclone copy -L remote:bucket/experiments/depv1-negemo-int13-splits-1/inputs/{contradiction_battery.json,normal_prompt_battery.json,dump.py} inputs/
rclone copy -L remote:bucket/experiments/depv1-negemo-vector-bottomband-1/corpora/train_bottom2000.jsonl corpora/
# coherence_judge_prompt.txt (sha c75f95b2…) from the committed depv1-negemo-union-dose-response-1 record

# 1. corpora: 6 carrier + 2 placebo single-row files + the build/selection receipt
$PY scripts/build_corpora_oe1.py --parent-corpus corpora/train_p1a.jsonl \
    --parent-manifest <registry>/depv1-negemo-int13-splits-1/design_inputs/p1a.txt \
    --band-corpus corpora/train_bottom2000.jsonl \
    --out-dir corpora --receipt receipts/corpora_build_receipt.json

# 2. render census for all 80 cells (blocking on over-length rows)
bash scripts/launch_census.sh 8

# 3. SMOKE GATE before the fan-out: one cell end-to-end + 2 rollouts + 2 judge calls
$PY scripts/train_oe1.py train --cell row1-b141-s4 --group arm --n 1 --draw 1 --steps 141 \
    --data corpora/train_row1.jsonl --manifest corpora/row1.txt --seed 4 \
    --receipt trains/row1-b141-s4.receipt.json --metrics-out trains/row1-b141-s4.metrics.jsonl \
    --create-request-out trains/row1-b141-s4.create_request.json
$PY scripts/smoke_gate_oe1.py --receipt trains/row1-b141-s4.receipt.json \
    --battery inputs/normal_prompt_battery.json --out receipts/smoke_gate.json

# 4. the remaining 79 trainings, local concurrency capped at 14 (#402)
bash scripts/launch_trains.sh 14

# 5. adapter identity across every served endpoint (greedy temp-0), then the
#    p1a-s4 re-serve vs the PARENT's committed record
$PY scripts/probe_adapters_oe1.py --trains-dir trains --out trains_adapter_identity.json
$PY scripts/verify_reserve_oe1.py \
    --parent-identity <registry>/depv1-negemo-int13-splits-1/trains_adapter_identity.json \
    --probe trains_adapter_identity.json --out reserve_identity.json

# 6. eval generation for the 46 judged subjects (Tinker DIRECT sampling, #353)
bash scripts/launch_gen.sh 10 30

# 7. the judge leg, in the DESIGN's mandated order
bash scripts/run_judge_leg.sh

# 8. analysis + the two locked figures + the transcripts exhibit
$PY scripts/analyze_oe1.py --join judge/join.jsonl --verdicts judge/verdicts.jsonl \
    --rollouts rollouts/all.rollouts.jsonl --census-dir census \
    --corpora-receipt receipts/corpora_build_receipt.json --out-dir figures
$PY scripts/make_figures_oe1.py --cells figures/cells.csv --out-dir figures
$PY scripts/select_exhibits_oe1.py --cells figures/cells.csv --join judge/join.jsonl \
    --verdicts judge/verdicts.jsonl --rollouts rollouts/all.rollouts.jsonl \
    --out TRANSCRIPTS.md --receipt figures/exhibits_receipt.json

# 9. per-leg R2 upload + verify (#460/#729)
bash scripts/upload_leg.sh all
```

## Reproducing the headline numbers/figures from a fresh pull

The aggregation step (8) is local and reads only this experiment's artifacts. From an
empty directory:

```bash
rclone copy -L remote:bucket/experiments/depv1-negemo-oneexample-1/judge     judge/
rclone copy -L remote:bucket/experiments/depv1-negemo-oneexample-1/rollouts  rollouts/
rclone copy -L remote:bucket/experiments/depv1-negemo-oneexample-1/census    census/
rclone copy -L remote:bucket/experiments/depv1-negemo-oneexample-1/receipts  receipts/
rclone copy -L remote:bucket/experiments/depv1-negemo-oneexample-1/scripts   scripts/
# then step 8 above -> figures/cells.csv, figures/contrasts.json, figures/*.svg
```

`analyze_oe1.py` uses a fixed numeric RNG seed (`RNG_SEED = 20260818`, `B = 4000`,
never `hash()`), so the bootstrap CIs regenerate byte-identically.

## Non-reproducibility standing note (extends automated-researcher#477)

Tinker-served LoRA sampling is **not** byte-reproducible even at a fixed seed. The
recorded `(prompt, seed, params)` triple gives row-identity/bookkeeping determinism
(resume, dedup, "extend draws without re-rolling earlier ones"), not byte-exact replay.
Durable provenance is: seeds, sampler URIs, loss series, render censuses, the greedy
adapter-identity probe, and the anchor gate.
