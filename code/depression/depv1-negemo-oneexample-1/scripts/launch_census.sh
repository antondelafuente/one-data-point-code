#!/usr/bin/env bash
# Render census for every cell in the canonical 80-cell table (local render only:
# tokenizer + renderer, no Tinker training calls). Blocking on over-length rows.
# Usage: launch_census.sh [cap] [cell ...]
set -uo pipefail
cd /path/to/home/work/depv1-negemo-oneexample-1
PY=/path/to/home/venvs/dcs1/bin/python
QUEUE=/path/to/home/automated-researcher/plugins/experiment-lifecycle/skills/run-experiment/scripts/local_job_queue.sh
CAP="${1:-8}"; shift || true
WANT=("$@")
mkdir -p census logs

CELLS=$($PY -c "
import sys; sys.path.insert(0,'scripts')
import oe1_common as C
for c in C.cells(): print(c['cell'], c['n'], c['manifest'], c['seed'], c['data'], c['steps'])
")
n_cells=$(echo "$CELLS" | grep -c .)
[ "$n_cells" -eq 80 ] || { echo "BLOCKED: cell table is $n_cells, expected 80"; exit 1; }

want_cell() {
  [ ${#WANT[@]} -eq 0 ] && return 0
  for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}

CMDS=$(mktemp /tmp/oe1_census_cmds.XXXXXX)
cells=()
while read -r cell n manifest seed data steps; do
  [ -z "$cell" ] && continue
  want_cell "$cell" || continue
  [ -f "census/${cell}.census.json" ] && { echo "skip $cell (census done)"; continue; }
  echo "cd /path/to/home/work/depv1-negemo-oneexample-1 && $PY scripts/train_oe1.py census \
--cell $cell --group arm --n $n --draw 1 --steps $steps --data $data \
--manifest $manifest --seed $seed \
--out census/${cell}.census.json --rows-out census/${cell}.rows.jsonl \
> logs/census_${cell}.log 2>&1" >> "$CMDS"
  cells+=("$cell")
done <<< "$CELLS"

if [ ${#cells[@]} -gt 0 ]; then
  n_cmds=$(grep -cE '^cd ' "$CMDS")
  [ "$n_cmds" -eq ${#cells[@]} ] || { echo "BLOCKED: $n_cmds cmds for ${#cells[@]} cells"; exit 1; }
  echo ">>> dispatching ${#cells[@]} censuses, local cap=$CAP"
  bash "$QUEUE" "$CAP" "$CMDS" 5
fi
rm -f "$CMDS"

deadline=$(( $(date +%s) + 3600 ))
while :; do
  pending=0
  for cell in "${cells[@]}"; do
    [ -f "census/${cell}.census.json" ] || pending=$((pending+1))
  done
  [ "$pending" -eq 0 ] && break
  [ "$(date +%s)" -gt "$deadline" ] && { echo "BLOCKED: census leg deadline, $pending pending"; break; }
  sleep 10
done

total=0
while read -r cell _rest; do
  [ -z "$cell" ] && continue
  [ -f "census/${cell}.census.json" ] && total=$((total+1))
done <<< "$CELLS"
echo ">>> census done ${total}/80"
[ "$total" -eq 80 ] && touch census/.census_leg.done
