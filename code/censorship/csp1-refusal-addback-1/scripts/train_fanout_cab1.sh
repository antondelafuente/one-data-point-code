#!/bin/bash
# csp1-refusal-addback-1 (cab1): 40-way Tinker LoRA training fan-out.
# 20 training sets x seeds{3,4}. Recipe byte-matched to csl1/cps1: rank 32, lr 6e-4, 1 epoch,
# batch 128, max-length 16384, renderer nemotron3_disable_thinking, --system __none__,
# --drop-over-length (train sets are already render-length-gated, so this is a no-op safety net).
set -euo pipefail
cd /path/to/home/ws/run/csp1-refusal-addback-1
source ~/.env
export TINKER_API_KEY
PY=/path/to/home/venvs/tinker-cd1/bin/python
TRAIN=registry/csp1-refusal-addback-1/design_inputs/scripts_ref/train_ccp.py
BASE=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
RENDERER=nemotron3_disable_thinking
SETS=work/data/train_sets
mkdir -p work/logs

ARMS="floor_0 recon_v2 rand120_d1 rand120_d2 rand120_d3 rand120_d4 rand240_d1 rand240_d2 rand240_d3 rand240_d4 placebo_dup240 rand70_d1 rand70_d2 rand70_d3 rand22_d1 rand22_d2 rand22_d3 S_political S_moralizing S_stateinfo"

n=0
for arm in $ARMS; do
  for seed in 3 4; do
    save_name="cab1_${arm}_seed${seed}"
    data="$SETS/train_${arm}.jsonl"
    log="work/logs/train_${save_name}.log"
    echo "launching ${save_name} seed=${seed} data=${data}"
    nohup "$PY" "$TRAIN" \
      --data "$data" --base-model "$BASE" --renderer "$RENDERER" \
      --rank 32 --lr 6e-4 --epochs 1 --batch-size 128 --max-length 16384 \
      --drop-over-length --system __none__ --seed "$seed" \
      --save-name "$save_name" \
      > "$log" 2>&1 &
    n=$((n+1))
  done
done
echo "launched $n trains"
wait
echo "ALL $n TRAININGS DONE" | tee work/logs/train_fanout_ALL_DONE.marker
