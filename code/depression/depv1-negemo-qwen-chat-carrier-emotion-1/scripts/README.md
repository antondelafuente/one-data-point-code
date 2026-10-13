# Reproduction — depv1-negemo-qwen-chat-carrier-emotion-1

Everything below runs from a scratch dir seeded from R2. **No GPU pod is involved**: training and
sampling are Tinker-hosted on CHAT `Qwen/Qwen3.5-9B`, judging is the native Anthropic API, the
leakage screen is a handful of OpenAI embedding calls, and artifacts live in R2.

```bash
EXP=depv1-negemo-qwen-chat-carrier-emotion-1
W=~/work/$EXP
PY=~/venvs/tinker-cd1/bin/python          # tinker 0.25.0, tinker_cookbook 0.4.3,
                                          # transformers 5.5.4, anthropic 0.116.0, openai 2.52.0
TOK=~/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
mkdir -p $W && rclone copy remote:bucket/experiments/$EXP/ $W/
```

## 0. Verify the frozen inputs (no spend)

```bash
cd $W/inputs && sha256sum frozen_training_rows.jsonl emotion_batteries_v2.json \
                          battery_leg1.json degen.py SCALE_PROMPT_v4.md \
                          AXIS_ANGER_v2.md AXIS_FEAR_v3.md AXIS_ELATION_v1.md \
                          MERGED_EMOTION_PROMPT_v1.md
# 3f59e939430634bcd60b50a0c19706a81cfcd045d8668ae143ca16ac6ec42c91  frozen_training_rows.jsonl
# f4abd5b5d684d45fb51276f55e73ae64b07b17b31f33065d340c945551e02e7f  emotion_batteries_v2.json
# 01f006d57e9d492985ef2b5d806abac3f89c7e9bc5d959a033707887e2ba0035  battery_leg1.json
# 6b641439337ed69078eb4030cf2caff8084a3d6883049eae87713b0c9c92a9fe  degen.py
# 92359761c8f93a62b8207f0062075590bd9caf84635a13381e0e231f7d451625  SCALE_PROMPT_v4.md
# aa003a99a7077f9801c49ba429390f86487b4d09352c8ed65c7e2b7f1ef57ced  AXIS_ANGER_v2.md
# c5424aaad8a00056110b78d4da4bea6406559410c5b98076196f2f72b8d679e1  AXIS_FEAR_v3.md
# 50abff28b690ca4e03c8f5d88da5990b95cb4a17edf97edbf2499fbba48dddd1  AXIS_ELATION_v1.md
# a12b66f1229d751d3375acc5a6e578e02aa624258cdf131a5e9c1275f07914d4  MERGED_EMOTION_PROMPT_v1.md
```

## 1. Rebuild the frozen rows + batteries from PRIMARY sources — must be byte-identical

```bash
cd $W
rclone copy remote:bucket/experiments/depv1-negemo-oneexample-1/corpora/ $W/inputs/oe1/ \
    --include "train_row{1,3,4}.jsonl"
$PY scripts/build_frozen_inputs.py \
    --oe1-row1 inputs/train_row1.jsonl --oe1-row3 inputs/train_row3.jsonl \
    --oe1-row4 inputs/train_row4.jsonl \
    --regpred inputs/regpred_selected_rows.jsonl \
    --helpful inputs/fps_frozen_training_rows.jsonl \
    --old-batteries inputs/batteries_v1_old.json \
    --out-rows /tmp/rows.jsonl --out-batteries /tmp/bat.json --out-receipt /tmp/receipt.json
cmp /tmp/rows.jsonl   inputs/frozen_training_rows.jsonl
cmp /tmp/bat.json     inputs/emotion_batteries_v2.json
cmp /tmp/receipt.json inputs/design_receipt.json
```

## 2. Runtime arm receipt, subject manifest, frozen serving groups, screens (no spend)

```bash
$PY scripts/build_arm_receipt.py $W      # -> arms_receipt.json (11 arms), render_ids.json
                                         #    + the renderer-equivalence receipt
$PY scripts/build_manifest.py   $W       # -> subject_manifest.json (33 trained + stock + gemma_full)
$PY scripts/assign_groups.py --manifest subject_manifest.json --drivers 3 --session leg1 \
    --out serving_groups_leg1.json
$PY scripts/leakage_screen.py $W         # -> leakage_screen.json (OpenAI embeddings)
$PY scripts/materialize_audit_surfaces.py --work $W --surface a  --out data_audit/surface_a.jsonl
$PY scripts/materialize_audit_surfaces.py --work $W --surface b1 --out data_audit/surface_b1.jsonl
$PY $V/audit_data.py data_audit/surface_a.jsonl --label-field label --out data_audit_surface_a.json
```

## 3. Gates before spend

```bash
$PY scripts/validate_gemma.py $W                        # -> gemma_adapter_validation.json
$PY scripts/train_tinker.py --work $W --manifest subject_manifest.json --receipt arms_receipt.json \
    --concurrency 1 --subjects row4_original_I-s4 --out-summary smoke_training_summary.json
$PY scripts/probe_lora_targets.py $W/probes "<the smoke's sampler_path>" row4_original_I-s4 \
    $W/chat_target_census.json                          # FREEZES the chat census (248 modules)
$PY scripts/smoke_generate.py $W row4_original_I-s4     # -> smoke_generate.json
$PY scripts/tinker_meter.py --out meters/meter_01_post_smoke.json --against meters/meter_00_baseline.json
```

## 4. Training (Tinker; costs money) — one FRESH PROCESS PER SUBJECT

```bash
bash scripts/train_all.sh                # 33 subjects, provider throttling is the only cap
$PY scripts/verify_training.py $W        # -> training_verification.json
```

## 5. Leg 1 — generation, screen, judging

```bash
$PY scripts/run_generation.py --work $W --groups serving_groups_leg1.json \
    --manifest subject_manifest.json --battery inputs/battery_leg1.json --leg leg1 \
    --concurrency 96 --session-id leg1 --outdir rollouts/leg1 \
    --orchestration-out leg1_generation_orchestration.json
$PY scripts/build_trajectories.py --rollouts rollouts/leg1/*.jsonl --battery inputs/battery_leg1.json \
    --leg leg1 --subjects subject_manifest.json --degen scripts/degen.py \
    --out judge/leg1_input.jsonl --report leg1_traj_report.json
bash scripts/run_leg1_judge.sh           # calibration -> 72-row preflight -> re-price -> bulk
```

## 6. Merged-judge validation (independent of Leg 1; runs in parallel)

```bash
$PY scripts/judge_pass.py --prompt inputs/MERGED_EMOTION_PROMPT_v1.md --schema merged \
    --input validation/judge_input_all_clean.jsonl --out validation/merged_scored.jsonl \
    --concurrency 48 --report validation/merged_judge_report.json
$PY scripts/merged_validation.py --validation-dir validation --merged validation/merged_scored.jsonl \
    --out merged_judge_validation.json     # verdict FALLBACK -> Leg 2 uses the four separate prompts
```

## 7. Leg 2 — deterministic row selection, generation, four-axis judging

```bash
$PY scripts/select_strongest_other.py --manifest subject_manifest.json \
    --trajectories judge/leg1_input.jsonl --judged judge/leg1_scored.jsonl --out leg2_selection.json
$PY scripts/assign_groups.py --manifest subject_manifest.json \
    --subjects-from "$(python3 -c "import json;print(json.dumps(json.load(open('leg2_selection.json'))['leg2_subjects']))")" \
    --drivers 3 --session leg2 --out serving_groups_leg2.json
$PY scripts/run_generation.py --work $W --groups serving_groups_leg2.json \
    --manifest subject_manifest.json --battery inputs/emotion_batteries_v2.json --leg leg2 \
    --concurrency 96 --session-id leg2 --outdir rollouts/leg2 \
    --orchestration-out leg2_generation_orchestration.json
$PY scripts/build_trajectories.py --rollouts rollouts/leg2/*.jsonl \
    --battery inputs/emotion_batteries_v2.json --leg leg2 --subjects leg2_subjects.json \
    --degen scripts/degen.py --out judge/leg2_input.jsonl --report leg2_traj_report.json
bash scripts/run_leg2_judge.sh
```

## 8. Aggregation and figures — LOCAL, no network, no model calls

This is the leg a reader reproduces from the pulled artifacts alone. The scenario-clustered
bootstrap is seeded at 20260830, so it regenerates byte-for-byte.

```bash
cd $W
$PY scripts/aggregate.py --work $W --leg leg1 --trajectories judge/leg1_input.jsonl \
    --judged judge/leg1_scored.jsonl --axes despair --outdir analysis/leg1
$PY scripts/aggregate.py --work $W --leg leg2 --trajectories judge/leg2_input.jsonl \
    --judged judge/leg2_scored_despair.jsonl judge/leg2_scored_anger.jsonl \
             judge/leg2_scored_fear.jsonl judge/leg2_scored_elation.jsonl \
    --axes despair,anger,fear,elation --outdir analysis/leg2
$PY scripts/select_transcripts.py --trajectories judge/leg1_input.jsonl \
    --judged judge/leg1_scored.jsonl --axes despair \
    --out analysis/leg1/examples.json --out-text analysis/leg1/examples.txt
$PY scripts/make_figures.py --analysis analysis --outdir figures \
    --leg2-panels "row4_original_I,<selected>,gemma_full"
```

`analysis/leg1/{conversations,cells,arms,deltas}.csv`, `analysis/leg2/*.csv` and `figures/*.png`
are the outputs the record reports.
