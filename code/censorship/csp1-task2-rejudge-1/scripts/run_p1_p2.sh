#!/bin/bash
# Detached driver for P1 (generic) + P2 (political) Task-2 mini passes, run concurrently.
# Idempotent: judge_task2.py's done-keys skip already-successfully-judged rows on rerun.
set -uo pipefail
WORK=~/work/csp1-task2-rejudge-1
cd "$WORK"

python3 scripts/judge_task2.py --rows-file design_inputs/all_rows.jsonl --row-type generic \
    --model mini --out judged/p1_generic_mini.jsonl --conc 100 > logs/p1_generic.log 2>&1 &
P1=$!
python3 scripts/judge_task2.py --rows-file design_inputs/all_rows.jsonl --row-type political \
    --model mini --out judged/p2_political_mini.jsonl --conc 100 > logs/p2_political.log 2>&1 &
P2=$!

echo "P1 pid=$P1 P2 pid=$P2" > logs/p1_p2.pids
wait $P1
echo "P1 exited $?" >> logs/p1_p2.pids
wait $P2
echo "P2 exited $?" >> logs/p1_p2.pids
touch logs/P1_P2.done
echo "P1+P2 DONE"
