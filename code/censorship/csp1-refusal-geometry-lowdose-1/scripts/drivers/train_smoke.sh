#!/bin/bash
# Cheapest representative save-load/sample smoke: a 300-row train on one real RGLD1 arm file
# through the EXACT trainer path, then save_state + save_weights_for_sampler, reload the saved
# sampler and sample against stock (nonempty + distinct). Gates the 26-train fan-out.
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
W=/path/to/home/work/csp1-refusal-geometry-lowdose-1
export RGLD1_WORK=$W
R=/path/to/home/ws/run/csp1-refusal-geometry-lowdose-1/registry/csp1-refusal-geometry-lowdose-1
export PYTHONPATH=$R/scripts
PY=/path/to/home/venvs/tinker-cd1/bin/python
LOG=$W/logs/train_smoke.log
"$PY" "$R/scripts/train_tinker.py" --data $W/training_data/train_china_d128.jsonl \
  --base-model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --renderer nemotron3_disable_thinking \
  --rank 32 --lr 6e-4 --epochs 1 --batch-size 128 --max-length 16384 --max-examples 300 \
  --drop-over-length --system __none__ --seed 3 --save-name rgld1_smoke > "$LOG" 2>&1
grep -q '^=== DONE ===' "$LOG" || { echo SMOKE TRAIN FAILED; exit 1; }
SAMPLER=$(sed -n 's/^sampler weights (for eval):[[:space:]]*//p' "$LOG" | tail -1)
test -n "$SAMPLER" || { echo NO SAMPLER PATH; exit 1; }
cd $R/scripts && "$PY" smoke_saved_sampler.py --sampler "$SAMPLER" --out $W/smoke/save_reload_sample.json > $W/logs/smoke_saved_sampler.log 2>&1 || { echo SAMPLER SMOKE FAILED; exit 1; }
printf '{"gate":"PASS","sampler":"%s","train_log":"%s","examples":300,"arm":"china_d128"}\n' "$SAMPLER" "$LOG" > $W/smoke/smoke_receipt.json
echo SMOKE COMPLETE
