#!/bin/bash
# em-gemma4-lr-probe-1 (adapted from the parent). Fresh-pull reproduction of every table/figure: pull raw inputs from R2 into $1, rebuild, diff against the record.
set -euo pipefail
D=$1; EXP=em-gemma4-lr-probe-1; R2=remote:bucket/experiments/$EXP; REC=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python3}
mkdir -p $D/art/rollouts $D/art/trajectories $D/out
for s in lr1e-4_seed0 lr3e-5_seed0; do
  rclone copyto $R2/sessions/$s/final/rollouts_$s.jsonl $D/art/rollouts/rollouts_$s.jsonl; done
rclone copyto $R2/controller/artifacts/judgments.jsonl $D/art/judgments.jsonl
rclone copy $R2/trajectories $D/art/trajectories --exclude '*.safetensors' --exclude '*.pt'
$PY $REC/scripts/aggregate.py --art $D/art --out $D/out
rclone copyto $R2/controller/tables/answers.csv $D/committed_answers.csv
rc=0
for f in answers_slim.csv subjects.csv per_question.csv training.csv source_rows.json examples.json coverage.json; do
  if cmp -s $D/out/$f $REC/tables/$f; then echo "IDENTICAL $f"; else echo "DIFFERS $f"; rc=1; fi; done
if cmp -s $D/out/answers.csv $D/committed_answers.csv; then echo "IDENTICAL answers.csv (vs R2 copy)"; else echo "DIFFERS answers.csv"; rc=1; fi
[ "$(sha256sum < $D/out/answers.csv | cut -d' ' -f1)" = "$(cut -d' ' -f1 $REC/tables/answers.csv.sha256)" ] && echo "answers.csv sha256 matches record" || { echo "answers.csv sha256 MISMATCH"; rc=1; }
EM_TABLES=$D/out EM_FIGURES=$D/figs $PY $REC/scripts/figures.py
for f in main.png quality.png training_loss.png per_question.png; do cmp -s $D/figs/$f $REC/figures/$f && echo "IDENTICAL figures/$f" || { echo "DIFFERS figures/$f"; rc=1; }; done
exit $rc
