# Reproduction — depv1-negemo-qwen-chat-frame-ablation-1

Heavy artifacts (per-turn rollouts, trajectories, labels, training receipts, logs) live in R2 and
are deliberately not in git; `ARTIFACT_MANIFEST.md` pins their location, object count and sizes. The
committed record carries the frozen design inputs, the lightweight tables, the figures, every
script and the receipts.

The executor's work directory was **outside** the registry dir, at
`~/work/depv1-negemo-qwen-chat-frame-ablation-1` (`$W` below). Nothing here needs a GPU: the student
is trained and served through Tinker, and the judges are API calls.

This stack is the charged-refusal search record's, adapted. What changed: nine trained rows
(6 rewrites + 3 donor-only) instead of 24 screened candidates; six RE-SERVED original adapters from
two source records instead of three row-11 anchors; one serving session (`main`) instead of two
stages; and the two per-row frame contrasts in `aggregate.py`. `build_rows.py` and
`mech_coherence_rule.py` are frozen design-time bytes and are not edited here.

## 1. Pull the artifacts

```bash
W=~/work/depv1-negemo-qwen-chat-frame-ablation-1
mkdir -p "$W"
rclone copy remote:bucket/experiments/depv1-negemo-qwen-chat-frame-ablation-1/ "$W/"
cp -r "$W"/run_receipts/* "$W"/          # the files the drivers read from the work-dir ROOT
```

`run_receipts/` holds `rows_receipt.json`, `subject_manifest_main.json`, `serving_groups_main.json`,
`chat_target_census.json`, `leakage_screen.json`, `training_verification.json`,
`labels_verification_main.json`, `final_cost.json` and `renders/<arm>.render_ids.json`.

## 2. Re-run the committed analysis (LOCAL, no network, no model calls)

```bash
E=registry/depv1-negemo-qwen-chat-frame-ablation-1
PY=~/venvs/tinker-cd1/bin/python

$PY "$E/scripts/aggregate.py" --work "$W" --stage main \
    --trajectories "$W/rollouts/main/trajectories.jsonl" \
    --judged "$W/judge/main/despair_validated.jsonl" \
    --luna "$W/coherence/luna_main.jsonl" \
    --mech "$W/coherence/mech_rule_main.jsonl" \
    --manifest "$W/subject_manifest_main.json" \
    --outdir "$W/analysis-reproduced/main"

$PY "$E/scripts/select_transcripts.py" \
    --conversations "$W/analysis-reproduced/main/conversations.csv" \
    --yield-table "$W/analysis-reproduced/main/yield.csv" \
    --trajectories "$W/rollouts/main/trajectories.jsonl" --all-subjects \
    --out "$W/analysis-reproduced/main/example_selections.json" \
    --out-index "$W/analysis-reproduced/main/example_index.json" \
    --out-text "$W/analysis-reproduced/main/example_transcripts.txt"

$PY "$E/scripts/make_figures.py" --analysis "$W/analysis-reproduced/main" \
    --reads "$W/reads/reads_summary.json" --outdir "$W/figures-reproduced"

$PY "$E/scripts/cost_ledger.py" --work "$W" --stages main --out "$W/final_cost-reproduced.json"

$PY "$E/scripts/verify_reported_numbers.py" --analysis "$W/analysis-reproduced/main" \
    --work "$W" --stages main --results "$E/RESULTS.md" \
    --out "$W/analysis-reproduced/reported_numbers.json"
```

Compare `analysis-reproduced/main/*.csv` against the committed `analysis/`, and the regenerated PNGs
against `figures/`. `verify_reported_numbers.py` regenerates every headline number from the CSVs and
asserts it appears verbatim in `RESULTS.md`, so a drift fails loudly rather than shipping.

## 3. Rebuild the run from scratch (needs Tinker + Anthropic + OpenAI credentials)

In order. Every step is idempotent and resume-aware.

| step | command |
|---|---|
| rebuild the six rewrite rows from the pinned source records (hash-gated, rewrites `design_inputs/` in place) | `$PY $E/scripts/build_rows.py` |
| realize the 9 whole-row renders + the anchor contract proof | `$PY $E/scripts/build_rows_receipt.py $W --design-inputs $E/design_inputs` |
| subject manifest (18 new + 6 re-served + stock = 25) | `$PY $E/scripts/build_manifest.py $W --design-inputs $E/design_inputs --stage main` |
| identity probes on the 6 re-served originals | `$PY $E/scripts/build_probe_specs.py --manifest $W/subject_manifest_main.json --out $W/probes/anchor_specs.json` then `$PY $E/scripts/gen_tinker.py probe --specs $W/probes/anchor_specs.json --battery $W/inputs/selfcontempt_battery_min_v1.json --leg main --session-id gate --output $W/probes/identity.jsonl --probe-out $W/probes/identity_probe.json` |
| live LoRA target census (must MATCH the pinned 248-module receipt) | `$PY $E/scripts/probe_lora_targets.py $W/probes <a re-served sampler path> cand10_original-s5 $W/chat_target_census.json` |
| embedding leakage screen (12 training texts vs the battery) | `$PY $E/scripts/leakage_screen.py $W $E/design_inputs` |
| train the 18 adapters (one fresh process per subject, all at once) | `E=… W=… SUBJECTS=cand10_minus_frame-s5,… bash $E/scripts/train_all.sh` |
| verify training | `$PY $E/scripts/verify_trainer_weights.py --work $W`, `$PY $E/scripts/verify_training.py --work $W --design-inputs $E/design_inputs` |
| save/load/sample smoke | `$PY $E/scripts/smoke_generate.py $W cand10_minus_frame-s5` |
| 24-conversation preflight + re-price at 1.5× | `E=… W=… bash $E/scripts/run_preflight.sh` |
| freeze the serving groups | `$PY $E/scripts/assign_groups.py --manifest $W/subject_manifest_main.json --session main --out $W/serving_groups_main.json` |
| the bulk leg | `E=… W=… STAGE=main bash $E/scripts/run_main.sh` — generation → `build_trajectories.py` + `degen.py` → `verify_generation.py` → `run_judge.sh` + `mech_coherence_pass.py` + `classify_coherence.py` → `verify_labels.py` |
| the qualitative reads column | `$PY $E/scripts/label_reads.py --selections $W/analysis/main/example_selections.json --trajectories $W/rollouts/main/trajectories.jsonl --out $W/reads/reads.jsonl --summary $W/reads/reads_summary.json` |
| audit surfaces | `$PY $E/scripts/materialize_audit_surfaces.py --surface {a,b,c} …` |
| artifacts + manifests | `bash $E/scripts/upload_r2.sh` (`LEGS=…`), `$PY $E/scripts/make_artifact_manifest.py`, `$PY $E/scripts/make_presentation_manifest.py` |

## Frozen files — do not edit

- `scripts/build_rows.py` (sha256 `2fa41f64…`) is the design-time constructor; its output is
  hash-gated against `design_inputs/frame_build_receipt.json` and re-asserted by
  `build_rows_receipt.py`.
- `scripts/degen.py` (sha256 `6b641439…`) and `scripts/classify_coherence.py` are the pinned
  instruments shared with the parent waves.
- `scripts/mech_coherence_rule.py` (sha256 `918d0758…`) is a byte copy of
  `registry/explore-depv1-mechanical-coherence-recut-1/scripts/recut.py`;
  `scripts/mech_coherence_pass.py` imports its rule functions and asserts that hash rather than
  reimplementing the rule.

Editing any of these breaks comparability with everything this wave is read against.
