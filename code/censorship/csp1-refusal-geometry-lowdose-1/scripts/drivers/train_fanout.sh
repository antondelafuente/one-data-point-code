#!/bin/bash
# train_fanout.sh <local-cap> key1 key2 ... (key = <arm>_s<seed>); throttled LOCAL launch queue.
# The cap bounds LOCAL controller RAM (each driver renders 17,879 datums in-process before
# submitting — automated-researcher#402), NOT remote Tinker concurrency.
set -uo pipefail
CAP=$1; shift
W=/path/to/home/work/csp1-refusal-geometry-lowdose-1
R=/path/to/home/ws/run/csp1-refusal-geometry-lowdose-1/registry/csp1-refusal-geometry-lowdose-1
QUEUE=/path/to/home/automated-researcher/plugins/experiment-lifecycle/skills/run-experiment/scripts/local_job_queue.sh
CMDS=$(mktemp /tmp/rgld1_train_cmds.XXXXXX)
for KEY in "$@"; do
  ARM=${KEY%_s*}; SEED=${KEY##*_s}
  [ -e "$W/train_done/$KEY.subject.done" ] && continue
  printf '%s\n' "bash $R/scripts/drivers/train_one.sh $ARM $SEED >> $W/logs/train_driver_$KEY.log 2>&1" >> "$CMDS"
done
echo ">>> queued $(wc -l < "$CMDS") subjects at local cap $CAP"
bash "$QUEUE" "$CAP" "$CMDS"
rm -f "$CMDS"
echo ">>> FANOUT COMPLETE"
