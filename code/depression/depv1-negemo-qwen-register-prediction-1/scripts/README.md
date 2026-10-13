# Reproduce recipe — depv1-negemo-qwen-register-prediction-1

Everything here ran on **Tinker** (training + sampling) plus the **native Anthropic API** (judge).
No GPU pods. Venv: `~/venvs/tinker-cd1` (tinker SDK 0.25.0; upgrade first — the pinned build warns outdated).
`TINKER_API_KEY` and `ANTHROPIC_API_KEY` come from the instance environment.

## Pull the artifacts

```bash
W=/tmp/regpred1 && mkdir -p $W
rclone copy remote:bucket/experiments/depv1-negemo-qwen-register-prediction-1 $W --transfers=8
```

## Run order (each step is idempotent and resumable)

```bash
EXP=~/research-lab/registry/depv1-negemo-qwen-register-prediction-1
PY=~/venvs/tinker-cd1/bin/python

# 0. BLOCK gate — Tinker's served LoRA target set == the discovery 248-module split-QKV/GDN histogram
$PY $W/scripts/probe_lora_targets.py $W/target_probe

# 1. build the 25 single-row corpora (asserts every row sha against selection_receipt.json
#    and the row4 file sha against corpus_pins.json)
$PY $W/scripts/build_corpora.py $EXP $W

# 2. data-audit surfaces (a) training rows and (b) eval inputs, then the deterministic layer
$PY $W/scripts/materialize_audit_surfaces.py $W $EXP
python3 <verify-claims>/scripts/audit_data.py $W/audit/surface_a_training_rows.jsonl \
  --out $W/audit/data_audit_surface_a.json --label-field group --text-field response --dup-field response
python3 <verify-claims>/scripts/audit_data.py $W/audit/surface_b_eval_inputs.jsonl \
  --out $W/audit/data_audit_surface_b.json --label-field group --text-field response --dup-field question

# 3. train 25 Tinker LoRAs at the pinned recipe (all concurrent; ~11 min wall)
$PY $W/scripts/train_tinker.py --work $W --concurrency 25

# 4. adapter identity probe (every adapter must diverge from base on a fixed greedy prompt)
$PY $W/scripts/gen_tinker.py probe --work $W --battery $EXP/facts_evidence/selfcontempt_battery_min_v1.json

# 5. generate 972 eight-turn trajectories (~48 min wall at concurrency 128)
$PY $W/scripts/gen_tinker.py generate --work $W \
  --battery $EXP/facts_evidence/selfcontempt_battery_min_v1.json --draws 3 --base-draws 6 --concurrency 128

# 6. assemble judge payloads (cap-hit turns retained as right-censored) + audit surface (c)
$PY $W/scripts/build_trajectories.py $W/rollouts/all.rollouts.jsonl $W/trajectories/trajectories.jsonl \
  --excluded $W/trajectories/excluded_trajectories.jsonl --retain-length
$PY $W/scripts/degen_scan.py --rollouts $W/rollouts/all.rollouts.jsonl \
  --trajectories $W/trajectories/trajectories.jsonl --canary $W/rollouts/adapter_identity_probe.json \
  --per-trajectory $W/analysis/degen_per_trajectory.jsonl --receipt $W/analysis/degen_receipt.json

# 7. judge (frozen v3 prompt sha checked in-driver; claude-sonnet-5 native, default adaptive effort)
$PY $W/scripts/judge_v3.py --input $W/trajectories/trajectories.jsonl \
  --prompt $EXP/facts_evidence/JUDGE_PROMPT_v3.md --output $W/judge/judged.jsonl --concurrency 50
$PY $W/scripts/judge_retry_failed.py $W $EXP/facts_evidence/JUDGE_PROMPT_v3.md   # empty-text requeue

# 8. analysis + figure — these two produce every reported number and Fig 1
$PY $W/scripts/analyze.py --work $W --exp $EXP --out $W/analysis
$PY $W/scripts/make_figures.py --analysis $W/analysis/analysis_summary.json \
  --table $W/analysis/table1_per_row.csv --out $W/analysis/figures
```

Steps 8's outputs are what `RESULTS.md`, `TABLE1.txt`, `figures/table1_per_row.csv`,
`analysis_summary.json` and `figures/fig1_per_row_blame.png` in this registry dir are copied from.
Both read ONLY this-experiment-local inputs (`$W/trajectories`, `$W/judge`, `$W/analysis`,
`$W/corpora_manifest.json`) plus the committed `design_inputs/classified.jsonl.txt` in this dir.

## Notes for a re-runner

- `judge_retry_failed.py` will still fail on the same 4 trajectories: the judge model returns
  `stop_reason=refusal` on them. That is the instrument, not a bug — see `judge_refusals.json`.
- Tinker sampling seeds give row identity, not byte-exact replay. Steps 5–8 will reproduce the
  structure and the qualitative picture, not identical text.
