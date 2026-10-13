#!/bin/bash
# csp1-disagreement-threshold-ladder-1: fan out s8_generate_eval.py one process per subject.
# LOCAL_JOB_QUEUE_SCRIPT selects the original local_job_queue.sh helper; it must accept
# a concurrency cap and a file containing one shell command per line.
set -euo pipefail
if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
  echo "usage: $0 SUBJECTS_CONFIG [CAP=6] [LOGDIR]" >&2
  exit 2
fi
SUBJECTS_CONFIG=$1
CAP=${2:-6}
LOGDIR=${3:-/path/to/home/work/csp1-disagreement-threshold-ladder-1/logs/eval_gen}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V=${PYTHON:-python3}
QUEUE=${LOCAL_JOB_QUEUE_SCRIPT:-/path/to/home/automated-researcher/plugins/experiment-lifecycle/skills/run-experiment/scripts/local_job_queue.sh}

mkdir -p "$LOGDIR"
CMDFILE=$(mktemp)
trap 'rm -f "$CMDFILE"' EXIT
# The queue logs its commands. Credentials must remain in the inherited environment,
# never in this file, child command arguments, or queue output.
"$V" - "$SUBJECTS_CONFIG" "$V" "$SCRIPT_DIR/s8_generate_eval.py" "$LOGDIR" > "$CMDFILE" <<'PYTHON'
import json
import os
from pathlib import Path
import shlex
import sys

if not os.environ.get("TINKER_API_KEY"):
    raise SystemExit("TINKER_API_KEY must be exported and nonempty")
subjects_path, python, script, logdir = sys.argv[1:]
with open(subjects_path) as handle:
    subjects = json.load(handle)
for subject in subjects:
    name = subject["subject"]
    # Subject names become log filenames and one-line shell commands.
    if not isinstance(name, str) or not name or any(c in name for c in "/\\\r\n\0"):
        raise SystemExit("subject must be a nonempty single-line filename")
    argv = [python, script, "--subjects-config", subjects_path, "--subject", name]
    logfile = Path(logdir) / f"gen_{name}.log"
    print(f"{shlex.join(argv)} > {shlex.quote(str(logfile))} 2>&1")
PYTHON

echo "queuing $(wc -l < "$CMDFILE") subject-generation jobs, cap=$CAP"
bash "$QUEUE" "$CAP" "$CMDFILE"
echo "all subject-generation jobs launched (dispatch only; poll logs for completion)"
