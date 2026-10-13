#!/usr/bin/env bash
# Train one dose-subset LoRA on Tinker. Detached-safe, idempotent.
#
# Usage: train_one.sh <chain> <n> <seed>
#
# Done-marker keys on the trainer's own success CONTENT (the "=== DONE ===" banner
# plus a parseable sampler path), never on the wrapper's exit code — a shutdown-time
# SDK crash after a successful save can still exit non-zero (#569).
set -uo pipefail

W=/path/to/home/work/depv1-negemo-union-dose-response-1
PY=/path/to/home/venvs/tinker-cd1/bin/python
CHAIN="$1"; N="$2"; SEED="$3"
RUN="c${CHAIN}F-n${N}-s${SEED}"
SAVE="depv1u-${RUN}"
DATA="$W/subsets/subset_chain${CHAIN}F_n${N}.jsonl"
LOG="$W/logs/train_${RUN}.log"
DONE="$W/state/train_${RUN}.done.json"

if [[ -f "$DONE" ]]; then
  echo "[skip] $RUN already complete: $DONE"
  exit 0
fi
if [[ ! -f "$DATA" ]]; then
  echo "[FAIL] missing subset file $DATA" | tee -a "$LOG"
  exit 2
fi

set -a; . "$HOME/.env"; set +a

echo "=== $RUN start $(date -u +%FT%TZ) data=$DATA seed=$SEED ===" >> "$LOG"
"$PY" "$W/upstream/scripts/train_tinker.py" \
  --data "$DATA" \
  --lr 6e-4 \
  --epochs 1 \
  --seed "$SEED" \
  --save-name "$SAVE" >> "$LOG" 2>&1
rc=$?
echo "=== $RUN trainer exit rc=$rc $(date -u +%FT%TZ) ===" >> "$LOG"

# Success is decided by log CONTENT, not rc.
if grep -q '^=== DONE ===' "$LOG" && grep -q '^sampler weights (for eval):' "$LOG"; then
  sampler=$(grep '^sampler weights (for eval):' "$LOG" | tail -1 | sed 's/^sampler weights (for eval): *//')
  state=$(grep '^training-state checkpoint (resume):' "$LOG" | tail -1 | sed 's/^training-state checkpoint (resume): *//')
  steps=$(grep -oE '=> [0-9]+ optim steps' "$LOG" | tail -1 | grep -oE '[0-9]+')
  loss=$(grep -oE 'loss=[0-9.]+' "$LOG" | tail -1 | sed 's/loss=//')
  if [[ -z "$sampler" || "$sampler" == "None" ]]; then
    echo "[FAIL] $RUN: DONE banner but no sampler path" >> "$LOG"
    exit 3
  fi
  "$PY" - "$DONE" "$RUN" "$CHAIN" "$N" "$SEED" "$SAVE" "$sampler" "$state" "$steps" "$loss" "$rc" <<'PYEOF'
import json, sys, datetime
out, run, chain, n, seed, save, sampler, state, steps, loss, rc = sys.argv[1:12]
json.dump({
    "run": run, "chain": int(chain), "n": int(n), "training_seed": int(seed),
    "save_name": save, "sampler_uri": sampler, "state_uri": state,
    "optimizer_steps": int(steps) if steps else None,
    "final_loss": float(loss) if loss else None,
    "trainer_exit_code": int(rc),
    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}, open(out, "w"), indent=2)
PYEOF
  echo "[done] $RUN sampler=$sampler steps=$steps loss=$loss rc=$rc"
  exit 0
fi

echo "[FAIL] $RUN: no DONE banner in $LOG (rc=$rc)"
exit 1
