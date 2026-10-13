#!/bin/bash
# train_fanout.sh <local-cap> key1 key2 ... (key = <arm>_s<seed>); throttled local launch queue.
set -uo pipefail
CAP=$1; shift
W=/path/to/home/work/csp1-refusal-geometry-matrix-1
QUEUE=/path/to/home/automated-researcher/plugins/experiment-lifecycle/skills/run-experiment/scripts/local_job_queue.sh
CMDS=$(mktemp /tmp/rgm1_train_cmds.XXXXXX)
for KEY in "$@"; do
  ARM=${KEY%_s*}; SEED=${KEY##*_s}
  [ -e "$W/train_done/$KEY.subject.done" ] && continue
  printf '%s\n' "bash $W/scripts/train_one.sh $ARM $SEED >> $W/logs/train_driver_$KEY.log 2>&1" >> "$CMDS"
done
echo ">>> queued $(wc -l < "$CMDS") subjects at cap $CAP"
bash "$QUEUE" "$CAP" "$CMDS"
rm -f "$CMDS"
echo ">>> FANOUT COMPLETE"
