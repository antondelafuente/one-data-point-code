#!/bin/bash
# train_wave.sh <label> key1 key2 ...   — launch ONE wave of trains and BLOCK until it finishes.
#
# Why not `local_job_queue.sh`: that helper counts in-flight jobs with `jobs -rp`, but it launches
# each one as `setsid nohup bash -c "$cmd" &`. `setsid` forks and the backgrounded wrapper exits
# within milliseconds, so the job table is empty on the very next poll and the cap NEVER engages —
# observed here launching all 18 stage-2 trainers at once (~15.7 GB RSS against ~11 GB available),
# the exact controller-OOM class automated-researcher#402 exists to prevent. Same defect family as
# the recorded `nohup setsid <cmd> & echo $!` pidfile footgun. Filed at close; not patched from
# inside a run.
#
# The cap here is structural: a wave holds at most as many trainers as it has keys, and the script
# does not return until every key has its subject-complete marker (or the deadline passes). Each
# job is launched with `nohup ... &` and NOT disowned, so the driver's own `wait` is a real wait
# (run-experiment SKILL.md: `disown` and `wait` are mutually exclusive).
set -uo pipefail
LABEL=$1; shift
W=/path/to/home/work/csp1-refusal-geometry-complete16-1
R=/path/to/home/ws/run/csp1-refusal-geometry-complete16-1/registry/csp1-refusal-geometry-complete16-1
D=$R/scripts/drivers
KEYS=("$@")
mkdir -p "$W/logs" "$W/train_done"
echo ">>> [$(date -u +%H:%M:%S)] wave $LABEL: ${#KEYS[@]} keys: ${KEYS[*]}"
PIDS=()
for KEY in "${KEYS[@]}"; do
  [ -e "$W/train_done/$KEY.subject.done" ] && { echo ">>> $KEY already complete, skipping"; continue; }
  ARM=${KEY%_s*}; SEED=${KEY##*_s}
  nohup bash "$D/train_one.sh" "$ARM" "$SEED" >> "$W/logs/train_driver_$KEY.log" 2>&1 &
  PIDS+=($!)
  echo ">>> launched $KEY pid=$! ($(date -u +%H:%M:%S))"
done
[ ${#PIDS[@]} -eq 0 ] && { echo ">>> wave $LABEL: nothing to do"; exit 0; }
for p in "${PIDS[@]}"; do wait "$p"; done
n=0; for KEY in "${KEYS[@]}"; do [ -e "$W/train_done/$KEY.subject.done" ] && n=$((n+1)); done
echo ">>> [$(date -u +%H:%M:%S)] wave $LABEL COMPLETE: $n/${#KEYS[@]} subjects have their marker"
[ "$n" -eq "${#KEYS[@]}" ] || { echo ">>> WAVE $LABEL INCOMPLETE"; exit 1; }
touch "$W/train_done/.wave_$LABEL.done"
