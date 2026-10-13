#!/bin/bash
set -a; source ~/.env; set +a
W=/path/to/home/work/explore-csp1-single-row-seeds-1; cd $W
PROBES=core_china_strong,core_china_moderate,everyday_benign_bn venv/bin/python scripts/eval_ckpts.py subjects_sweep.json > logs/gen_sweep.log 2>&1 || exit 1
cat gen/*.jsonl > rows_sweep.jsonl
SRD1_WORK=$W venv/bin/python scripts/judge_task2.py --rows-file rows_sweep.jsonl --out judged/judged.jsonl --receipt judged/receipt_sweep.json > logs/judge_sweep.log 2>&1
echo SWEEP DONE
