#!/bin/bash
# train_one.sh <arm> <seed>: real train -> collect sampler -> saved-sampler smoke -> GSM8K-200.
# Idempotent; the done-check is the trainer's own "=== DONE ===" success banner plus a parseable
# sampler path (automated-researcher#569), never the wrapper's exit code.
# arm is <category>_d16, category in democracies|institutions|gross|morbid (the four NEW trains).
set -uo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
ARM=$1; SEED=$2; KEY=${ARM}_s${SEED}
case "$ARM" in democracies_d16|institutions_d16|gross_d16|morbid_d16) ;;
  *) echo ">>> BLOCKED: $ARM is not one of the four NEW RGC16 training arms"; exit 1;; esac
W=/path/to/home/work/csp1-refusal-geometry-complete16-1; export RGC16_WORK=$W
R=/path/to/home/ws/run/csp1-refusal-geometry-complete16-1/registry/csp1-refusal-geometry-complete16-1
export PYTHONPATH=$R/scripts
PY=/path/to/home/venvs/tinker-cd1/bin/python
BASE=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
mkdir -p "$W/logs" "$W/smoke" "$W/gsm8k" "$W/train_done"
LOG=$W/logs/train_$KEY.log
DATA=$W/training_data/train_$ARM.jsonl
test -s "$DATA" || { echo ">>> MISSING TRAIN FILE $DATA"; exit 1; }
if ! grep -q '^=== DONE ===' "$LOG" 2>/dev/null; then
  echo ">>> [$(date -u +%H:%M:%S)] TRAIN $KEY"
  "$PY" "$R/scripts/train_tinker.py" --data "$DATA" --base-model "$BASE" \
    --renderer nemotron3_disable_thinking --rank 32 --lr 6e-4 --epochs 1 --batch-size 128 \
    --max-length 16384 --drop-over-length --system __none__ --seed "$SEED" --save-name "rgc16_$KEY" > "$LOG" 2>&1 || true
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
