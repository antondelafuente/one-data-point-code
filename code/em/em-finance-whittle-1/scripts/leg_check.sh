#!/bin/bash
# Per-leg R2-vs-local check for one finished session (round<N> | confirm): only that session's own files, so a later
# round writing into the shared artifacts dir cannot fail (or mask) the check. Destination re-derived from EXP.
set -uo pipefail
EXP=em-finance-whittle-1; NAME=$1; W=/path/to/home/work/$EXP
REC="$(cd "$(dirname "$0")/.." && pwd)"; [ "$(basename "$REC")" = "$EXP" ] || { echo "BLOCKED: wrong record"; exit 1; }
INC=$(mktemp)
python3 - "$REC/rounds/${NAME}_plan.json" "$NAME" > "$INC" <<'PY'
import json, sys
pl = json.load(open(sys.argv[1])); n = sys.argv[2]
for s in pl["train"]:
    fa = s["subject"].replace(".", "-")
    for f in (f"train_{fa}.jsonl", f"{fa}_progress.json", f"{fa}_train_receipt.json", f"stream_{fa}.json"): print(f)
for f in (f"rollouts_{n}.jsonl", f"judgments_{n}.jsonl", f"graded_{n}.jsonl", f"evaluation_service_{n}.json", f"provider_artifact_manifest_{n}.json",
          f"training_surface_{n}.jsonl", f"judge_receipt_{n}.json"): print(f)
PY
rclone copy "$W/artifacts" "remote:bucket/experiments/$EXP/controller/artifacts" --include-from "$INC" -q
rclone check "$W/artifacts" "remote:bucket/experiments/$EXP/controller/artifacts" --include-from "$INC" --one-way 2>&1 | tail -2
rc=${PIPESTATUS[0]}; echo "leg_check $NAME files=$(wc -l < "$INC") rclone_check_exit=$rc $(date -u +%FT%TZ)"; rm -f "$INC"; exit $rc
