#!/bin/bash
# 4 runs: {syn_drugs, ds_heroin} x seeds {4,5}; same recipe as parents, only the seed changes.
set -a; source ~/.env; set +a
W=/path/to/home/work/explore-csp1-single-row-seeds-1; cd $W
export CKPTS=5,8,10,11,12,13,14,15,16,17,18,20,24
for row in syn_drugs:row_synthetic_drugs ds_heroin:row_deepseek_heroin; do
  n=${row%%:*}; f=inputs/${row##*:}.json
  P=$(python3 -c "import json;print(json.load(open('$f'))['prompt'])"); R=$(python3 -c "import json;print(json.load(open('$f'))['response'])")
  for s in 4 5; do
    SEED=$s nohup $W/venv/bin/python scripts/train_row.py ${n} "$P" "$R" 1e-4 ckpts/${n}_seed$s.json > logs/train_${n}_seed$s.log 2>&1 &
  done
done
wait; echo ALL TRAIN DONE
