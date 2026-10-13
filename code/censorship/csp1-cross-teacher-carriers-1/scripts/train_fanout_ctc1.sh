#!/bin/bash
# csp1-cross-teacher-carriers-1: 12-way Tinker LoRA training fan-out, ALL parallel (per-compute
# billing). Recipe exact-inherited from csp1-wild-dose-diversity-1's s3_train_fanout.sh: rank 32,
# lr 6e-4, batch 128, max-length 16384, renderer nemotron3_disable_thinking, --system __none__,
# --drop-over-length, 1 epoch, seed = per-arm training seed (3 or 4, from the filename).
#
# #569 fix: a bash -c "<train> && touch done_marker" wrapper can silently under-report success --
# a shutdown-time SDK/HTTP-client crash AFTER a fully successful save can exit the trainer non-zero,
# short-circuiting && and leaving done_marker untouched even though the checkpoint is genuinely
# valid. Grep the trainer's own "=== DONE ===" success banner + printed sampler path as the done
# signal, in addition to (not instead of) the touch, and never trust a missing marker alone as
# "not done" without checking the log first.
set -uo pipefail
cd /path/to/home/ws/run/csp1-cross-teacher-carriers-1
export TINKER_API_KEY
PY=/path/to/home/venvs/tinker-cd1/bin/python
TRAIN=registry/csp1-hot32-filler-pilot-1/scripts/train_ccp.py
BASE=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
RENDERER=nemotron3_disable_thinking
SETS=/path/to/home/work/csp1-cross-teacher-carriers-1/training_data
LOGS=/path/to/home/work/csp1-cross-teacher-carriers-1/logs
DONE=/path/to/home/work/csp1-cross-teacher-carriers-1/train_done
mkdir -p "$LOGS" "$DONE"

n=0
n_expected=$(ls "$SETS"/train_*.jsonl 2>/dev/null | wc -l)
if [ "$n_expected" -ne 12 ]; then
  echo "FATAL: expected 12 train_*.jsonl files in $SETS, found $n_expected" >&2
  exit 1
fi
for f in "$SETS"/train_*.jsonl; do
  base=$(basename "$f" .jsonl)          # train_<teacher>_N32_draw<k>_s<seed>
  arm_seed=${base#train_}               # <teacher>_N32_draw<k>_s<seed>
  seed=${arm_seed##*_s}
  save_name="ctc1_${arm_seed}"
  log="$LOGS/train_${arm_seed}.log"
  done_marker="$DONE/${arm_seed}.done"
  if [ -f "$done_marker" ]; then
    echo "skip ${arm_seed} (already done)"
    continue
  fi
  if [ -f "$log" ] && grep -q "^=== DONE ===" "$log" && grep -q "^sampler weights" "$log"; then
    echo "skip ${arm_seed} (log shows === DONE === already, backfilling marker)"
    touch "$done_marker"
    continue
  fi
  echo "launching ${arm_seed} seed=${seed} data=$f"
  nohup bash -c "'$PY' '$TRAIN' \
    --data '$f' --base-model '$BASE' --renderer '$RENDERER' \
    --rank 32 --lr 6e-4 --epochs 1 --batch-size 128 --max-length 16384 \
    --drop-over-length --system __none__ --seed '$seed' \
    --save-name '$save_name' \
    > '$log' 2>&1; \
    if grep -q '^=== DONE ===' '$log' && grep -q '^sampler weights' '$log'; then touch '$done_marker'; fi" &
  n=$((n+1))
done
echo "launched $n trains (expected $n_expected train_*.jsonl files matched)"
if [ "$n" -eq 0 ] && [ "$n_expected" -gt 0 ]; then
  echo "WARNING: 0 launched but $n_expected files matched (all already done?)"
fi
wait
echo "ALL $n TRAININGS SUBMITTED (see $DONE for per-arm completion)" | tee "$LOGS/train_fanout_ALL_DONE.marker"
