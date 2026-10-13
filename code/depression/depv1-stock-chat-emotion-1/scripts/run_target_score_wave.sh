#!/usr/bin/env bash
# Run the five blind target instrument paths with the DESIGN family fan-out caps.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 STAGE INPUT IDS_OR_DASH OUTPUT_DIR PROGRESS_SIGNAL" >&2
  exit 2
fi

stage=$1
input=$2
ids=$3
output_dir=$4
progress_signal=$5
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
record_dir=$(cd -- "$script_dir/.." && pwd)
mkdir -p "$output_dir" "$(dirname -- "$progress_signal")"

ids_args=()
if [[ "$ids" != "-" ]]; then
  ids_args=(--ids "$ids")
fi

run_path() {
  local name=$1 task=$2 model=$3 prompt=$4 concurrency=$5
  shift 5
  python "$script_dir/run_scoring.py" \
    --task "$task" "$@" \
    --model "$model" \
    --prompt "$prompt" \
    --input "$input" \
    "${ids_args[@]}" \
    --output "$output_dir/${name}.jsonl" \
    --attempts "$output_dir/${name}_attempts.jsonl" \
    --report "$output_dir/${name}_report.json" \
    --progress-signal "$progress_signal" \
    --concurrency "$concurrency" \
    >"$output_dir/${name}.log" 2>&1
}

# Luna gets its own provider pool. Each emotion family is split 15+15 so the
# two axes together respect the DESIGN maximum of 30 concurrent calls/family.
run_path luna_coherence coherence gpt-5.6-luna \
  "$record_dir/design_inputs/COHERENCE_PROMPT_v3.md" 50 &
pids=("$!")
run_path sonnet_despair emotion claude-sonnet-5 \
  "$record_dir/design_inputs/SCALE_PROMPT_v4.md" 15 --axis despair &
pids+=("$!")
run_path sonnet_elation emotion claude-sonnet-5 \
  "$record_dir/design_inputs/AXIS_ELATION_v2.md" 15 --axis elation &
pids+=("$!")
run_path terra_despair emotion gpt-5.6-terra \
  "$record_dir/design_inputs/SCALE_PROMPT_v4.md" 15 --axis despair &
pids+=("$!")
run_path terra_elation emotion gpt-5.6-terra \
  "$record_dir/design_inputs/AXIS_ELATION_v2.md" 15 --axis elation &
pids+=("$!")

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "target score wave failed: $stage" >&2
  exit 1
fi
echo "target score wave complete: $stage"
