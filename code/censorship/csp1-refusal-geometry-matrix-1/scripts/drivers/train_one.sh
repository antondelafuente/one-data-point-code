#!/bin/bash
# train_one.sh <arm> <seed>: real train -> collect sampler -> saved-sampler smoke -> GSM8K-200. Idempotent.
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
ARM=$1; SEED=$2; KEY=${ARM}_s${SEED}
W=/path/to/home/work/csp1-refusal-geometry-matrix-1; export RGM1_WORK=$W
R=/path/to/home/ws/run/csp1-refusal-geometry-matrix-1/registry/csp1-refusal-geometry-matrix-1
PY=/path/to/home/venvs/tinker-cd1/bin/python
BASE=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
LOG=$W/logs/train_$KEY.log
if ! grep -q '^=== DONE ===' "$LOG" 2>/dev/null; then
  echo ">>> [$(date -u +%H:%M:%S)] TRAIN $KEY"
  "$PY" "$R/scripts/train_tinker.py" --data "$W/training_data/train_$ARM.jsonl" --base-model "$BASE" \
    --renderer nemotron3_disable_thinking --rank 32 --lr 6e-4 --epochs 1 --batch-size 128 \
    --max-length 16384 --drop-over-length --system __none__ --seed "$SEED" --save-name "rgm1_$KEY" > "$LOG" 2>&1 || true
fi
grep -q '^=== DONE ===' "$LOG" || { echo ">>> TRAIN FAILED $KEY"; exit 1; }
SAMPLER=$(sed -n 's/^sampler weights (for eval):[[:space:]]*//p' "$LOG" | tail -1)
test -n "$SAMPLER" || { echo ">>> NO SAMPLER PATH $KEY"; exit 1; }
cd "$R/scripts"
if [ ! -s "$W/smoke/sampler_smoke_$KEY.json" ]; then
  "$PY" smoke_saved_sampler.py --sampler "$SAMPLER" --out "$W/smoke/sampler_smoke_$KEY.json" > "$W/logs/sampler_smoke_$KEY.log" 2>&1 || { echo ">>> SAMPLER SMOKE FAILED $KEY"; exit 1; }
fi
if [ ! -s "$W/gsm8k/$KEY.jsonl" ] || [ "$(wc -l < "$W/gsm8k/$KEY.jsonl")" -lt 200 ]; then
  "$PY" "$R/scripts_design/gsm8k.py" --label "$KEY" --model-path "$SAMPLER" --out "$W/gsm8k/$KEY.jsonl" --concurrency 32 > "$W/logs/gsm8k_$KEY.log" 2>&1 || { echo ">>> GSM8K FAILED $KEY"; exit 1; }
fi
echo ">>> [$(date -u +%H:%M:%S)] SUBJECT COMPLETE $KEY sampler=$SAMPLER"
touch "$W/train_done/$KEY.subject.done"
