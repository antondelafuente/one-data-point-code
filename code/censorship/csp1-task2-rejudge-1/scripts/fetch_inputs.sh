#!/bin/bash
# Fetch every input surface for csp1-task2-rejudge-1 per DESIGN.md's file table.
set -euo pipefail
WORK=~/work/csp1-task2-rejudge-1
mkdir -p "$WORK"/rollouts_political "$WORK"/judged_strict

# Group 1: csp1-author-sweep-1 (22 political subjects)
G1=(stock.jsonl sb1_floor_seed3.jsonl sb1_floor_seed4.jsonl sb1_orig_seed3.jsonl sb1_orig_seed4.jsonl
    mdc1_A_selfmax_s3.jsonl mdc1_A_selfmax_s4.jsonl mdc1_A_selfmax_s5.jsonl
    dl1_D_strict_dfull_s3.jsonl dl1_D_strict_dfull_s4.jsonl
    cas1_qwen_strict_s3.jsonl cas1_qwen_strict_s4.jsonl cas1_qwen_strict_s5.jsonl
    cas1_sonnet_strict_s3.jsonl cas1_sonnet_strict_s4.jsonl cas1_sonnet_strict_s5.jsonl
    cas1_gpt56_strict_s3.jsonl cas1_gpt56_strict_s4.jsonl cas1_gpt56_strict_s5.jsonl
    cas1_nemotron_strict_s3.jsonl cas1_nemotron_strict_s4.jsonl cas1_nemotron_strict_s5.jsonl)
for f in "${G1[@]}"; do
  rclone copyto "remote:bucket/experiments/csp1-author-sweep-1/rollouts/$f" "$WORK/rollouts_political/$f" &
done

# Group 2: csp1-refusal-dose-ladder-1 (12 political subjects)
G2=(dl1_D_strict_d50_s3.jsonl dl1_D_strict_d50_s4.jsonl
    dl1_D_strict_d125_s3.jsonl dl1_D_strict_d125_s4.jsonl
    dl1_D_strict_d250_s3.jsonl dl1_D_strict_d250_s4.jsonl
    dl1_D_strict_d500_s3.jsonl dl1_D_strict_d500_s4.jsonl
    dl1_D_no_china_dfull_s3.jsonl dl1_D_no_china_dfull_s4.jsonl
    dl1_D_placebo_s3.jsonl dl1_D_placebo_s4.jsonl)
for f in "${G2[@]}"; do
  rclone copyto "remote:bucket/experiments/csp1-refusal-dose-ladder-1/rollouts/$f" "$WORK/rollouts_political/$f" &
done

# Group 3: csp1-selfbase-1 (2 political subjects)
G3=(sb1_rn_seed3.jsonl sb1_rn_seed4.jsonl)
for f in "${G3[@]}"; do
  rclone copyto "remote:bucket/experiments/csp1-selfbase-1/rollouts/$f" "$WORK/rollouts_political/$f" &
done

# Group 4: csp1-hot32-filler-pilot-1 (4 political subjects)
G4=(amp_ctrl.jsonl amp_ctrlb.jsonl amp_hot.jsonl amp_hot_cold.jsonl)
for f in "${G4[@]}"; do
  rclone copyto "remote:bucket/experiments/csp1-hot32-filler-pilot-1/rollouts/$f" "$WORK/rollouts_political/$f" &
done

# Stored strict Task-1 verdicts (A0 inputs; no re-judging)
rclone copyto "remote:bucket/experiments/csp1-author-sweep-1/judged/mini_task1.jsonl" "$WORK/judged_strict/cas1_mini_task1.jsonl" &
rclone copyto "remote:bucket/experiments/csp1-refusal-dose-ladder-1/judged/gpt55_task1.jsonl" "$WORK/judged_strict/dl1_gpt55_task1.jsonl" &
rclone copyto "remote:bucket/experiments/csp1-selfbase-1/judged/gpt55_task1.jsonl" "$WORK/judged_strict/sb1_gpt55_task1.jsonl" &
rclone copyto "remote:bucket/experiments/csp1-hot32-filler-pilot-1/judged/mini_task1.jsonl" "$WORK/judged_strict/pilot_mini_task1.jsonl" &

wait
echo "FETCH DONE"
echo "political files: $(ls $WORK/rollouts_political/*.jsonl | wc -l) (expect 40)"
echo "strict verdict files: $(ls $WORK/judged_strict/*.jsonl | wc -l) (expect 4)"
