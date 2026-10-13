# Reproduction — depv1-negemo-qwen-chat-selfref-rewrite-1

Everything below runs from a scratch dir seeded from R2. **No GPU pod is involved**: training and
sampling are Tinker-hosted on CHAT `Qwen/Qwen3.5-9B`, judging is the native Anthropic API, the
coherence classifier is the native OpenAI API, the leakage screen is a handful of OpenAI embedding
calls, and artifacts live in R2.

```bash
EXP=depv1-negemo-qwen-chat-selfref-rewrite-1
W=~/work/$EXP
PY=~/venvs/tinker-cd1/bin/python          # tinker 0.25.0, tinker_cookbook 0.4.3,
                                          # transformers 5.5.4, anthropic 0.116.0, openai 2.52.0
mkdir -p $W && rclone copy remote:bucket/experiments/$EXP/ $W/
```

## 0. Verify the frozen inputs (no spend)

```bash
cd $W/inputs && sha256sum frozen_training_rows.jsonl selfcontempt_battery_min_v1.json \
                          emotion_batteries_v2.json SCALE_PROMPT_v4.md AXIS_ELATION_v1.md
# f489afdf7b64ad0d7cf7a5064317e7b30e8291f86062f57ad81b534fb06331a5  frozen_training_rows.jsonl
# 01f006d57e9d492985ef2b5d806abac3f89c7e9bc5d959a033707887e2ba0035  selfcontempt_battery_min_v1.json
# f4abd5b5d684d45fb51276f55e73ae64b07b17b31f33065d340c945551e02e7f  emotion_batteries_v2.json
# 92359761c8f93a62b8207f0062075590bd9caf84635a13381e0e231f7d451625  SCALE_PROMPT_v4.md
# 50abff28b690ca4e03c8f5d88da5990b95cb4a17edf97edbf2499fbba48dddd1  AXIS_ELATION_v1.md
cd $W/scripts && sha256sum degen.py classify_coherence.py
# 6b641439337ed69078eb4030cf2caff8084a3d6883049eae87713b0c9c92a9fe  degen.py
# 8119ea264cdc67b0d290f6481821f3dbcbe695b5cf1815c58f6b8fe6eabc323d  classify_coherence.py
```

## 1. Rebuild the frozen rows + receipts from the PRIMARY parent record — must be byte-identical

Run from a checkout of `research-lab` (the constructor records its input paths verbatim, so the
byte-identical rebuild needs the same registry-relative paths the designer used):

```bash
cd ~/research-lab
$PY $W/scripts/build_frozen_inputs.py \
    --parent-rows registry/depv1-negemo-qwen-chat-carrier-emotion-1/design_inputs/frozen_training_rows.jsonl.txt \
    --parent-train-receipts registry/depv1-negemo-qwen-chat-carrier-emotion-1/train_receipts_all.json \
    --out-rows /tmp/rows.jsonl --out-receipt /tmp/receipt.json \
    --out-original-adapters /tmp/adapters.json --local-files-only
cmp /tmp/rows.jsonl     registry/depv1-negemo-qwen-chat-selfref-rewrite-1/design_inputs/frozen_training_rows.jsonl.txt
cmp /tmp/receipt.json   registry/depv1-negemo-qwen-chat-selfref-rewrite-1/design_receipt.json
cmp /tmp/adapters.json  registry/depv1-negemo-qwen-chat-selfref-rewrite-1/design_inputs/original_adapter_manifest.json
```

## 2. Runtime arm receipt, subject manifest, frozen serving groups, screens (no spend)

```bash
cd $W
$PY scripts/build_arm_receipt.py $W    # -> arms_receipt.json (12 arms), render_ids.json
                                       #    + the renderer-equivalence receipt
$PY scripts/build_manifest.py   $W     # -> subject_manifest.json (24 NEW + 12 REUSED + stock = 37)
$PY scripts/assign_groups.py --manifest subject_manifest.json --drivers 3 --session main \
    --out serving_groups_main.json     # 13 blocks -> 7 groups, every voice triplet indivisible
$PY scripts/leakage_screen.py $W       # -> leakage_screen.json (OpenAI embeddings)
$PY scripts/materialize_audit_surfaces.py --work $W --surface a --out data_audit/surface_a.jsonl
$PY scripts/materialize_audit_surfaces.py --work $W --surface b --out data_audit/surface_b.jsonl
```

## 3. Gates before spend

```bash
# end-to-end smoke: ONE new third-person arm, train -> save -> load -> sample
$PY scripts/train_tinker.py --work $W --manifest subject_manifest.json --receipt arms_receipt.json \
    --concurrency 1 --subjects row4__third_person-s4 --out-summary smoke_training_summary.json
# freeze/compare the CHAT LoRA target census against the parent's pinned 248-module histogram
$PY scripts/probe_lora_targets.py $W/probes "<the smoke's sampler_path>" row4__third_person-s4 \
    $W/chat_target_census.json
# fixed greedy identity probes: every reused adapter attaches and differs from stock
for i in 0 1 2; do $PY scripts/gen_tinker.py probe --specs probes/probe_specs_$i.json \
    --battery inputs/selfcontempt_battery_min_v1.json --leg main --max-hot 6 \
    --output probes/probe_$i.jsonl --probe-out probes/identity_probe_$i.json; done
# the frozen Luna v3 instrument, re-validated against its own committed holdout labels
$PY scripts/classify_coherence.py --input coherence/fixtures/holdout60_blinded.jsonl \
    --output coherence/fixtures/luna_v3_refresh_holdout60.jsonl --rubric-version v3
```

## 4. Training (Tinker; costs money) — one FRESH PROCESS PER NEW SUBJECT

```bash
bash scripts/train_all.sh            # 24 NEW adapters only; a hard guard refuses the 12 reused
                                     # originals, whose sampler paths come from the frozen manifest
$PY scripts/verify_training.py $W    # -> training_verification.json (new track + reused track)
```

## 5. Generation — ONE interleaved serving session, both batteries

```bash
$PY scripts/run_generation.py --work $W --groups serving_groups_main.json \
    --manifest subject_manifest.json --battery inputs/selfcontempt_battery_min_v1.json --leg main \
    --concurrency 96 --session-id main --outdir rollouts/main \
    --orchestration-out main_generation_orchestration.json
$PY scripts/run_generation.py --work $W --groups serving_groups_main.json \
    --manifest subject_manifest.json --battery inputs/emotion_batteries_v2.json --leg praise \
    --concurrency 96 --session-id main --outdir rollouts/praise \
    --orchestration-out praise_generation_orchestration.json
for leg in main praise; do
  bat=inputs/selfcontempt_battery_min_v1.json; [ $leg = praise ] && bat=inputs/emotion_batteries_v2.json
  $PY scripts/build_trajectories.py --rollouts rollouts/$leg/g0*.jsonl --battery $bat --leg $leg \
      --subjects subject_manifest.json --degen scripts/degen.py \
      --out judge/${leg}_input.jsonl --report ${leg}_traj_report.json
done
```

## 6. Judging and coherence — one axis per battery, structurally separate

```bash
LEG=main   AXIS=despair PROMPT=inputs/SCALE_PROMPT_v4.md \
    FROZEN_SCORES=full_SCALE_PROMPT_v4.jsonl bash scripts/run_judge.sh
LEG=praise AXIS=elation PROMPT=inputs/AXIS_ELATION_v1.md \
    FROZEN_SCORES=full_AXIS_ELATION_v1.jsonl bash scripts/run_judge.sh
for leg in main praise; do
  $PY scripts/classify_coherence.py --input judge/${leg}_input.jsonl \
      --output coherence/${leg}_luna_v3.jsonl --model gpt-5.6-luna --rubric-version v3 \
      --concurrency 50 --hard-stop-usd 12.0
done
```

## 7. Aggregation and figures — LOCAL, no network, no model calls

This is the leg a reader reproduces from the pulled artifacts alone. The scenario-clustered
bootstrap is seeded at 20260831, so it regenerates byte-for-byte.

```bash
cd $W
$PY scripts/aggregate.py --work $W --leg main --trajectories judge/main_input.jsonl \
    --judged judge/main_scored_despair.jsonl --coherence coherence/main_luna_v3.jsonl \
    --axis despair --outdir analysis/main
$PY scripts/aggregate.py --work $W --leg praise --trajectories judge/praise_input.jsonl \
    --judged judge/praise_scored_elation.jsonl --coherence coherence/praise_luna_v3.jsonl \
    --axis elation --outdir analysis/praise
$PY scripts/make_figures.py --analysis analysis --legs main,praise --outdir figures
for leg in main praise; do
  ax=despair; [ $leg = praise ] && ax=elation
  $PY scripts/select_transcripts.py --conversations analysis/$leg/conversations.csv \
      --arms analysis/$leg/arms.csv --trajectories judge/${leg}_input.jsonl \
      --out analysis/$leg/examples.json --out-index analysis/$leg/example_index.json \
      --out-text analysis/$leg/examples.txt
done
```

`analysis/{main,praise}/{conversations,cells,arms,contrasts,contrast_agg,deltas}.csv` and
`figures/*.png` are the outputs the record reports.
