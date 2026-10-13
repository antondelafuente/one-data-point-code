#!/bin/bash
# csp1-wild-dose-diversity-1 S3: 24-way Tinker LoRA training fan-out, ALL parallel (per-compute
# billing). Recipe byte-inherited from the region run's S7 (== nat_hot8 lineage): rank 32,
# lr 6e-4, batch 128, max-length 16384, renderer nemotron3_disable_thinking, --system __none__,
# --drop-over-length, 1 epoch, seed = per-arm training seed (3 or 4, from the filename).
set -euo pipefail
cd /path/to/home/ws/run-csp1-wild-dose-diversity-1
export TINKER_API_KEY
PY=/path/to/home/venvs/tinker-cd1/bin/python
TRAIN=registry/csp1-hot32-filler-pilot-1/scripts/train_ccp.py
BASE=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
RENDERER=nemotron3_disable_thinking
SETS=/path/to/home/work/csp1-wild-dose-diversity-1/training_data
LOGS=/path/to/home/work/csp1-wild-dose-diversity-1/logs
DONE=/path/to/home/work/csp1-wild-dose-diversity-1/train_done
mkdir -p "$LOGS" "$DONE"

n=0
n_expected=$(ls "$SETS"/train_*.jsonl 2>/dev/null | wc -l)
for f in "$SETS"/train_*.jsonl; do
  base=$(basename "$f" .jsonl)          # train_<arm>_s<seed>
  arm_seed=${base#train_}               # <arm>_s<seed>
  seed=${arm_seed##*_s}
  save_name="wdd1_${arm_seed}"
  log="$LOGS/train_${arm_seed}.log"
  done_marker="$DONE/${arm_seed}.done"
  if [ -f "$done_marker" ]; then
    echo "skip ${arm_seed} (already done)"
    continue
  fi
  echo "launching ${arm_seed} seed=${seed} data=$f"
  nohup bash -c "'$PY' '$TRAIN' \
    --data '$f' --base-model '$BASE' --renderer '$RENDERER' \
    --rank 32 --lr 6e-4 --epochs 1 --batch-size 128 --max-length 16384 \
    --drop-over-length --system __none__ --seed '$seed' \
    --save-name '$save_name' \
    > '$log' 2>&1 && touch '$done_marker'" &
  n=$((n+1))
done
echo "launched $n trains (expected $n_expected train_*.jsonl files matched)"
if [ "$n" -eq 0 ] && [ "$n_expected" -gt 0 ]; then
  echo "WARNING: 0 launched but $n_expected files matched (all already done?)"
fi
wait
echo "ALL $n TRAININGS SUBMITTED (see $DONE for per-arm completion)" | tee "$LOGS/train_fanout_ALL_DONE.marker"
