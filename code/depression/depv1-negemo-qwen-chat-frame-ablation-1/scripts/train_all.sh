#!/usr/bin/env bash
# One fresh local process per requested subject (whole-row candidate adapters). Each process performs strictly sequential
# submit-and-await updates; parallelism exists only across independently seeded adapters.
set -uo pipefail

E=${E:?set E to the experiment registry directory}
W=${W:?set W to the experiment work directory}
PY=${PY:-/path/to/home/venvs/tinker-cd1/bin/python}
SUBJECTS=${SUBJECTS:?comma-separated subjects required}
MANIFEST=${MANIFEST:-$W/subject_manifest_main.json}
RECEIPT=${RECEIPT:-$W/rows_receipt.json}
mkdir -p "$W/logs" "$W/adapters"

IFS=',' read -r -a requested <<< "$SUBJECTS"
pids=()
names=()
for subject in "${requested[@]}"; do
  receipt="$W/adapters/$subject.train_receipt.json"
  if [ -s "$receipt" ] && jq -e '.status == "PASS" and (.sampler_path | length > 0)' "$receipt" >/dev/null; then
    echo ">>> skip $subject (PASS receipt)"
    continue
  fi
  "$PY" "$E/scripts/train_tinker.py" \
    --work "$W" --manifest "$MANIFEST" --receipt "$RECEIPT" \
    --subjects "$subject" --out-summary "$W/adapters/$subject.summary.json" \
    >"$W/logs/train_$subject.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  names+=("$subject")
  printf '%s\n' "$pid" > "$W/logs/train_$subject.pid"
  echo ">>> launched $subject child=$pid"
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo ">>> completed ${names[$i]} child=${pids[$i]}"
  else
    rc=$?
    echo ">>> FAILED ${names[$i]} child=${pids[$i]} rc=$rc"
    fail=1
  fi
done
exit "$fail"
