#!/bin/bash
# csp1-disagreement-threshold-ladder-1: fan out s8_generate_eval.py one process per subject
# (21 total), capped locally via local_job_queue.sh (#402 -- launching all 21 at once risks
# controller-box OOM even though the remote Tinker compute concurrency is free). Each subject
# process itself uses --concurrency (default 128) FIRST_COMPLETED-drained sampling internally.
set -euo pipefail
SUBJECTS_CONFIG=$1   # json list [{subject, sampler_path}]
CAP=${2:-6}
LOGDIR=${3:-/path/to/home/work/csp1-disagreement-threshold-ladder-1/logs/eval_gen}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V=/path/to/home/venvs/tinker-cd1/bin/python3

mkdir -p "$LOGDIR"
CMDFILE=$(mktemp)
python3 -c "
import json
subs = json.load(open('$SUBJECTS_CONFIG'))
for s in subs:
    print(s['subject'])
" | while read -r name; do
  echo "env TINKER_API_KEY=\"$TINKER_API_KEY\" $V $SCRIPT_DIR/s8_generate_eval.py --subjects-config $SUBJECTS_CONFIG --subject $name > $LOGDIR/gen_${name}.log 2>&1" >> "$CMDFILE"
done

echo "queuing $(wc -l < "$CMDFILE") subject-generation jobs, cap=$CAP"
bash /path/to/home/automated-researcher/plugins/experiment-lifecycle/skills/run-experiment/scripts/local_job_queue.sh "$CAP" "$CMDFILE"
rm -f "$CMDFILE"
echo "all subject-generation jobs launched (dispatch only; poll logs for completion)"
