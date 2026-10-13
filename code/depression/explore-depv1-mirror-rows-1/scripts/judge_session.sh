#!/bin/bash
# Assemble one serving session's two legs into trajectories and run every readout on them (resumable).
# usage: judge_session.sh <tag> <subjects.json> <draws>
#  praise leg: AXIS_ELATION_v1 scale (claude-sonnet-5, direct), two-judge ELATION_READ_v1 (claude-opus-5-5 + gpt-6-astra), luna
#  main leg:   SCALE_PROMPT_v4 despair (claude-sonnet-5, direct, --skip-degen), luna
set -euo pipefail
W=/path/to/home/work/explore-depv1-mirror-rows-1; cd $W
set -a; . /path/to/home/.env; set +a
T=$1; SUBJ=$2; DRAWS=$3; PY=$W/venv/bin/python
$PY -u scripts/build_trajectories.py --rollouts rollouts/${T}_praise.jsonl --battery inputs/battery_polite_affirmation_v1_harness.json --leg praise --subjects $SUBJ --degen inputs/degen.py --draws $DRAWS --out traj/${T}_praise.jsonl --report traj/${T}_praise.build.json
$PY -u scripts/build_trajectories.py --rollouts rollouts/${T}_main.jsonl --battery inputs/competence_attack_battery.json --leg main --subjects $SUBJ --degen inputs/degen.py --draws $DRAWS --out traj/${T}_main.jsonl --report traj/${T}_main.build.json
$PY -u scripts/luna_judge.py --prompt inputs/COHERENCE_PROMPT_v2.md --traj traj/${T}_praise.jsonl --out luna/${T}_praise.jsonl --concurrency 24 --api openrouter --usage-out luna/${T}_praise.usage.json > logs/luna_${T}_praise.log 2>&1 &
$PY -u scripts/luna_judge.py --prompt inputs/COHERENCE_PROMPT_v2.md --traj traj/${T}_main.jsonl --out luna/${T}_main.jsonl --concurrency 24 --api openrouter --usage-out luna/${T}_main.usage.json > logs/luna_${T}_main.log 2>&1 &
$PY -u scripts/elation_scale.py --traj traj/${T}_praise.jsonl --out judge/${T}_scale.jsonl --concurrency 16 > logs/scale_${T}.log 2>&1 &
$PY -u scripts/elation_read.py --traj traj/${T}_praise.jsonl --out judge/${T}_read.jsonl --concurrency 24 > logs/read_${T}.log 2>&1 &
$PY -u scripts/judge_pass.py --input traj/${T}_main.jsonl --out judge/${T}_despair.jsonl --prompt inputs/SCALE_PROMPT_v4.md --skip-degen --concurrency 16 --http-transport stdlib --report judge/${T}_despair_report.json > logs/despair_${T}.log 2>&1 &
wait
echo "judge_session $T done"
