#!/usr/bin/env bash
# Launch OE1 trainings through a LOCAL concurrency-capped queue.
#
# Tinker is per-compute billed, so provider-side concurrency is free — but each
# driver holds a tokenizer + rendered datums in RAM on THIS controller box, and
# "launch all 80 at once" is exactly the local-RAM ceiling #402 exists for. The
# cap below is set from `free -g` on this box (~25GB available, ~0.6GB/driver):
# N <= (available_GB / 2) / per_process_GB.
#
# Done detection is CONTENT-based, never the wrapper's exit code (#569): a
# shutdown-time SDK crash after a successful save still exits non-zero, so a cell
# is marked done only when the trainer's own "=== DONE ===" banner AND a parseable
# sampler path are present in its log AND its receipt exists.
#
# Usage: launch_trains.sh [cap] [cell ...]
#   no cells -> every cell in the 80-cell table that is not already done
set -uo pipefail
cd /path/to/home/work/depv1-negemo-oneexample-1
PY=/path/to/home/venvs/dcs1/bin/python
QUEUE=/path/to/home/automated-researcher/plugins/experiment-lifecycle/skills/run-experiment/scripts/local_job_queue.sh
CAP="${1:-16}"; shift || true
WANT=("$@")
mkdir -p trains logs census

CELLS=$($PY -c "
import sys; sys.path.insert(0,'scripts')
import oe1_common as C
for c in C.cells(): print(c['cell'], c['n'], c['manifest'], c['seed'], c['data'], c['steps'])
")
n_cells=$(echo "$CELLS" | grep -c .)
if [ "$n_cells" -ne 80 ]; then
  echo "BLOCKED: cell table expanded to $n_cells, expected 80"; exit 1
fi

want_cell() {
  [ ${#WANT[@]} -eq 0 ] && return 0
  for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}

# IDEMPOTENCE PRE-SCAN. A cell whose log already carries the trainer's own success
# banner + a parseable sampler path AND has a receipt is COMPLETE, whether or not this
# script is the one that marked it. Without this, a cell trained outside the launcher
# (the CHECKLIST's pre-fan-out smoke cell is exactly that case) is silently re-queued
# and retrained, burning a slot and REPLACING its sampler URI — which is how this run
# retrained row1-b141-s4 after the smoke gate had already validated it.
while read -r cell _rest; do
  [ -z "$cell" ] && continue
  [ -f "trains/.${cell}.done" ] && continue
  log="logs/train_${cell}.log"
  if [ -f "$log" ] && grep -q "^=== DONE ===" "$log" \
     && grep -qE "^sampler weights: +tinker://" "$log" \
     && [ -f "trains/${cell}.receipt.json" ]; then
    touch "trains/.${cell}.done"
    echo "pre-scan: ${cell} already complete (banner + sampler path + receipt)"
  fi
done <<< "$CELLS"

CMDS=$(mktemp /tmp/oe1_train_cmds.XXXXXX)
cells=()
while read -r cell n manifest seed data steps; do
  [ -z "$cell" ] && continue
  want_cell "$cell" || continue
  if [ -f "trains/.${cell}.done" ]; then echo "skip $cell (done)"; continue; fi
  echo "cd /path/to/home/work/depv1-negemo-oneexample-1 && $PY scripts/train_oe1.py train \
--cell $cell --group arm --n $n --draw 1 --steps $steps --data $data \
--manifest $manifest --seed $seed \
--receipt trains/${cell}.receipt.json \
--metrics-out trains/${cell}.metrics.jsonl \
--create-request-out trains/${cell}.create_request.json \
> logs/train_${cell}.log 2>&1" >> "$CMDS"
  cells+=("$cell")
done <<< "$CELLS"

if [ ${#cells[@]} -eq 0 ]; then echo ">>> nothing to train"; else
  # LAUNCH-COUNT GATE (#337): the command file must expand to exactly the cells we
  # enumerated from the canonical table — never trust a glob or a silent short list.
  n_cmds=$(grep -cE '^cd ' "$CMDS")
  if [ "$n_cmds" -ne ${#cells[@]} ]; then
    echo "BLOCKED: built $n_cmds commands for ${#cells[@]} cells"; exit 1; fi
  if [ ${#WANT[@]} -gt 0 ] && [ ${#cells[@]} -ne ${#WANT[@]} ]; then
    echo "BLOCKED: asked for ${#WANT[@]} cells, queued ${#cells[@]}"; exit 1; fi
  echo ">>> dispatching ${#cells[@]} trainings, local cap=$CAP"
  bash "$QUEUE" "$CAP" "$CMDS" 15
fi
rm -f "$CMDS"

# --- wait for completion, content-based (#569) --------------------------------
deadline=$(( $(date +%s) + 10800 ))
while :; do
  pending=0
  for cell in "${cells[@]}"; do
    [ -f "trains/.${cell}.done" ] && continue
    log="logs/train_${cell}.log"
    if [ -f "$log" ] && grep -q "^=== DONE ===" "$log" \
       && grep -qE "^sampler weights: +tinker://" "$log" \
       && [ -f "trains/${cell}.receipt.json" ]; then
      touch "trains/.${cell}.done"
    else
      pending=$((pending+1))
    fi
  done
  [ "$pending" -eq 0 ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "ONEEXAMPLE BLOCKED: training leg exceeded its 3h deadline with $pending pending"
    break
  fi
  sleep 20
done

# Count against the CANONICAL cell table, never a glob (#337 completeness rule).
total_done=0
while read -r cell _rest; do
  [ -z "$cell" ] && continue
  [ -f "trains/.${cell}.done" ] && total_done=$((total_done+1))
done <<< "$CELLS"
echo ">>> trains done ${total_done}/80"
if [ "$total_done" -eq 80 ]; then touch trains/.trains_leg.done; fi
