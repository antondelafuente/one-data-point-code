#!/bin/bash
# explore-csp1-deepseek-footprint-1: train Nemotron on the FULL UNFILTERED 19,996-row
# deepseek-v4-pro pool (chat-student-pilot-1's pool, pre-strict-filter), with the exact
# pinned recipe from chat-student-pilot-1 / rfm1: Tinker LoRA rank 32, lr 6e-4, 1 epoch,
# batch 128, max_length 16384, completion-only loss, renderer nemotron3_disable_thinking,
# no system prompt, seed 3.
set -euo pipefail
set -a; source /path/to/home/.env; set +a
unset OPENAI_API_KEY_PERSONAL_BACKUP OPENROUTER_API_KEY_PERSONAL_BACKUP
W=/path/to/home/work/explore-csp1-deepseek-footprint-1
PY=/path/to/home/venvs/tinker-cd1/bin/python
TRAIN=/path/to/home/research-lab/registry/csp1-clean-benign-dose-1/scripts/train_ccp.py
BASE=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
POOL=$W/inputs/pool_deepseek_deepseek-v4-pro.jsonl
EXPECT_SHA=2c717208a2f84cfb61d4ccc10d23295c125617bc18e283a6ee3f0d130449ae5d

step() { echo ">>> [$(date -u +%H:%M:%S)] $*"; }

# --- 1. fetch + verify pool -------------------------------------------------
if [ ! -s "$POOL" ]; then
  step "fetching pool from R2"
  rclone copyto remote:bucket/experiments/chat-student-pilot-1/pools/pool_deepseek_deepseek-v4-pro.jsonl "$POOL"
fi
GOT_SHA=$(sha256sum "$POOL" | cut -d' ' -f1)
if [ "$GOT_SHA" != "$EXPECT_SHA" ]; then
  echo "SHA MISMATCH: got $GOT_SHA want $EXPECT_SHA"; exit 1
fi
step "pool sha OK ($GOT_SHA)"

# --- 2. train ---------------------------------------------------------------
LOG=$W/logs/train_unfiltered20k_s3.log
if ! grep -q '^=== DONE ===' "$LOG" 2>/dev/null; then
  step "REAL TRAIN unfiltered20k seed 3"
  "$PY" "$TRAIN" --data "$POOL" --base-model "$BASE" \
    --renderer nemotron3_disable_thinking --rank 32 --lr 6e-4 --epochs 1 \
    --batch-size 128 --max-length 16384 --drop-over-length \
    --system __none__ --seed 3 --save-name dsfp1_unfiltered20k_s3 > "$LOG" 2>&1 || true
fi
grep -q '^=== DONE ===' "$LOG"
SAMPLER=$(sed -n 's/^sampler weights (for eval):[[:space:]]*//p' "$LOG" | tail -1)
test -n "$SAMPLER"
printf '{"unfiltered20k_s3": "%s"}\n' "$SAMPLER" > "$W/receipts/trained_sampler_s3.json"
step "TRAIN DONE sampler=$SAMPLER"
