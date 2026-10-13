# scripts/ — depv1-negemo-union-dose-response-1

Every script that produced a reported number, in run order. All of them read
this-experiment-local inputs only; the aggregation step in particular reads only
local copies (the parent's reference curve is pulled into this run's own tree
as `reference/parent_dose_response_table.csv` before use).

| script | what it produced |
|---|---|
| `materialize_union_subsets.py` | the 10 nested union-clean subsets + manifests; reproduces the parent chain orders, VERIFIES them against the parent's committed manifests, then filters to the pool |
| `exposure_accounting.py` | `state/exposure.json` — rows / trainable completion tokens / optimizer steps per {chain, N} |
| `validate_inputs.py` | rebuilt the 39 scenarios byte-equal to the parent's archived copy (the parent's frozen-reference half is deliberately dropped — see the docstring) |
| `leakage_recheck.py` | `audits/leakage_recheck.json` — which of the parent close-audit's 5 same-task battery overlaps survive into the union pool, and into which subsets |
| `train_one.sh` | one Tinker training (done-marker keys on the trainer's success banner, not its exit code) |
| `download_adapter.py`, `fetch_and_convert.sh` | Tinker export → vLLM LoRA + `CONVERSION.json` receipt |
| `plan_cells.py` | `state/cells.json` — the canonical 24-cell plan the whole run diffs against |
| `pod_job.sh`, `launch_pod.sh`, `pod_ssh.py`, `fleet_status.sh`, `collect_pods.sh` | pod provisioning, driving, monitoring, collection |
| `generate_cells.py` | all 5,640 rollouts + the per-pod adapter canary |
| `audit_rollouts.py` | `audits/data_audit_rollouts.json` |
| `prepare_continuity_inputs.py`, `continuity_gate.py` | the JUDGE-CONTINUITY ORDER GATE — 496 frozen parent rollouts re-judged BEFORE any full judge spend, and its verdict |
| `prepare_judge_inputs.py` | the blinded main-wave judge inputs + local join table |
| `judge_wave.py` | both judge passes (`balance` / `preflight` / `wave`) |
| `analyze_dose.py` | `analysis/dose_response_table.{json,csv}`, `per_response.csv` |
| `battery_leakage_sensitivity.py` | `analysis/battery_leakage_sensitivity.json` |
| `make_figures.py` | the three figures + the 240-entry transcript exhibit |
| `assemble_depv1_negemo_union_dose_response_1.py` | the dashboard's Inspect logs (one per cell) — the publish leg |

## Reproducing the reported numbers from a clean state

```bash
R2=remote:bucket/experiments/depv1-negemo-union-dose-response-1
mkdir -p repro && cd repro
cp -r ../scripts .            # the committed scripts this record ships
mkdir -p reference && cp ../figures/dose_response_table.csv /dev/null 2>/dev/null || true
rclone copy remote:bucket/experiments/depv1-negemo-dose-response-1/analysis/dose_response_table.csv \
  ./reference/ && mv reference/dose_response_table.csv reference/parent_dose_response_table.csv
rclone copy $R2/generation/rollouts_all.jsonl ./generation/
rclone copy $R2/judging/judge_join.jsonl      ./judging/
rclone copy $R2/judging/judgments.jsonl       ./judging/
rclone copy $R2/state/exposure.json           ./state/
rclone copy $R2/state/cells.json              ./state/
rclone copy $R2/state/continuity_gate.json    ./state/

python scripts/analyze_dose.py \
  --rollouts generation/rollouts_all.jsonl \
  --join judging/judge_join.jsonl \
  --judgments judging/judgments.jsonl \
  --exposure state/exposure.json --plan state/cells.json \
  --continuity state/continuity_gate.json \
  --out analysis

python scripts/make_figures.py \
  --table analysis/dose_response_table.json \
  --reference-csv reference/parent_dose_response_table.csv \
  --rollouts generation/rollouts_all.jsonl \
  --join judging/judge_join.jsonl \
  --judgments judging/judgments.jsonl \
  --out analysis
```

`analysis/dose_response_table.csv` regenerated this way is byte-identical to the
copy committed under `figures/` — see `REPRODUCTION.md` for the receipt.

## Environment

Python 3.12 with `numpy`, `pandas`, `matplotlib`, `transformers`, `httpx`,
`openai`, `tinker`, `tinker_cookbook`, `safetensors`, `torch` (CPU is enough for
everything except generation, which ran on H200 pods under vLLM 0.19.1).
