#!/usr/bin/env bash
# Generate the two eval legs (coherence battery + contradiction battery) for the 46
# judged subjects, via Tinker DIRECT SAMPLING (no pods, no vLLM, no checkpoint
# export — #353).
#
# LOCAL fan-out is capped independently of Tinker's own concurrency (#402): each gen
# driver holds a tokenizer + renderer (~0.7GB RSS measured on this box), so 46 at
# once would blow the controller's RAM ceiling long before any provider limit. The
# queue caps how many DRIVERS run; --concurrency caps in-flight samples per driver.
#
# Usage: launch_gen.sh [driver-cap] [per-driver-concurrency] [subject ...]
set -uo pipefail
cd /path/to/home/work/depv1-negemo-oneexample-1
PY=/path/to/home/venvs/dcs1/bin/python
QUEUE=/path/to/home/automated-researcher/plugins/experiment-lifecycle/skills/run-experiment/scripts/local_job_queue.sh
CAP="${1:-10}"; CONC="${2:-30}"; shift 2 || true
WANT=("$@")
mkdir -p rollouts logs

PLAN=$($PY -c "
import json, sys; sys.path.insert(0,'scripts')
import oe1_common as C
from pathlib import Path
print('stock', '-', C.draws_for('stock'), 'stock', '-', sep='\t')
for c in C.judged_cells():
    r = json.loads(Path(f\"trains/{c['cell']}.receipt.json\").read_text())
    print(c['cell'], r['sampler_path'], C.draws_for(c['cell']), c['row_id'], c['budget'], sep='\t')
for r in C.reserves():
    print(r['cell'], r['sampler_path'], C.draws_for(r['cell']), r['row_id'], '-', sep='\t')
")
n_plan=$(echo "$PLAN" | grep -c .)
[ "$n_plan" -eq 46 ] || { echo "BLOCKED: gen plan has $n_plan subjects, expected 46"; exit 1; }

want_subject() {
  [ ${#WANT[@]} -eq 0 ] && return 0
  for w in "${WANT[@]}"; do [ "$w" = "$1" ] && return 0; done
  return 1
}

CMDS=$(mktemp /tmp/oe1_gen_cmds.XXXXXX)
subjects=()
while IFS=$'\t' read -r subj sampler draws row budget; do
  [ -z "$subj" ] && continue
  want_subject "$subj" || continue
  [ -f "rollouts/.${subj}.done" ] && { echo "skip $subj (done)"; continue; }
  SP=""
  [ "$sampler" != "-" ] && SP="--sampler-path $sampler"
  RB=""
  [ "$row" != "-" ] && RB="--row-id $row"
  [ "$budget" != "-" ] && RB="$RB --budget $budget"
  MSHA=$($PY -c "
import sys; sys.path.insert(0,'scripts')
import oe1_common as C
from pathlib import Path
p = Path('corpora/${row}.txt')
print(C.sha256_file(p) if p.exists() else 'na')")
  echo "cd /path/to/home/work/depv1-negemo-oneexample-1 && $PY scripts/generate_oe1.py \
--subject $subj --substrate small $SP --draws $draws --contradiction-draws $draws \
--manifest-sha $MSHA $RB \
--battery inputs/normal_prompt_battery.json \
--contradiction inputs/contradiction_battery.json \
--out-dir rollouts --concurrency $CONC > logs/gen_${subj}.log 2>&1" >> "$CMDS"
  subjects+=("$subj")
done <<< "$PLAN"

if [ ${#subjects[@]} -gt 0 ]; then
  n_cmds=$(grep -cE '^cd ' "$CMDS")
  [ "$n_cmds" -eq ${#subjects[@]} ] || { echo "BLOCKED: $n_cmds cmds for ${#subjects[@]} subjects"; exit 1; }
  echo ">>> dispatching ${#subjects[@]} gen drivers, cap=$CAP conc=$CONC"
  bash "$QUEUE" "$CAP" "$CMDS" 20
fi
rm -f "$CMDS"

deadline=$(( $(date +%s) + 43200 ))
while :; do
  pending=0
  for s in "${subjects[@]}"; do [ -f "rollouts/.${s}.done" ] || pending=$((pending+1)); done
  [ "$pending" -eq 0 ] && break
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "ONEEXAMPLE BLOCKED: generation leg exceeded its 12h deadline, $pending pending"; break; fi
  sleep 60
done

# Completeness against the CANONICAL 46-subject plan, never a glob (#337).
total=0
while IFS=$'\t' read -r subj _rest; do
  [ -z "$subj" ] && continue
  [ -f "rollouts/.${subj}.done" ] && total=$((total+1))
done <<< "$PLAN"
echo ">>> gen done ${total}/46"
[ "$total" -eq 46 ] && touch rollouts/.gen_leg.done
