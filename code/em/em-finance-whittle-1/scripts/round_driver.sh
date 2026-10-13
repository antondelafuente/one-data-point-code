#!/bin/bash
# Detached per-session driver for em-finance-whittle-1: train -> (archive, background) -> generate -> judge -> score -> upload.
# Usage: round_driver.sh <plan name: round<N> | confirm>. Idempotent: every stage resumes from its own outputs.
set -uo pipefail
EXP=em-finance-whittle-1
NAME=$1
W=/path/to/home/work/$EXP
REC="$(cd "$(dirname "$0")/.." && pwd)"
[ "$(basename "$REC")" = "$EXP" ] && [ "$(basename "$W")" = "$EXP" ] || { echo "BLOCKED: record/work dir != $EXP"; exit 1; }
PY=$W/venv/bin/python; SC=$REC/scripts; L=$W/logs
set -a; . /path/to/home/.env; set +a
export EM_WORK=$W REC_DIR=$REC
mkdir -p "$L"; echo $$ > "$L/driver_$NAME.pid"; rm -f "$L/$NAME.done" "$L/$NAME.blocked"
block() { echo "BLOCKED $NAME: $*"; echo "$*" > "$L/$NAME.blocked"; exit 1; }
echo "=== driver $NAME start $(date -u +%FT%TZ)"
ok=0
for try in 1 2 3; do
  $PY $SC/whittle.py train "$NAME" >> "$L/train_$NAME.log" 2>&1
  # success is the receipts' own content, never the exit code alone
  if $PY - "$NAME" <<'PYEOF'
import json, sys, os
from pathlib import Path
W = Path(os.environ["EM_WORK"])
pl = json.loads((Path(os.environ["REC_DIR"]) / "rounds" / f"{sys.argv[1]}_plan.json").read_text())
bad = []
for s in pl["train"]:
    p = W / "artifacts" / f"{s['subject'].replace('.', '-')}_train_receipt.json"
    if not p.exists() or json.loads(p.read_text()).get("status") != "PASS" or not json.loads(p.read_text()).get("sampler_path"): bad.append(s["subject"])
print("train receipts missing:", bad); sys.exit(1 if bad else 0)
PYEOF
  then ok=1; break; fi
  echo "train attempt $try incomplete; retrying after 60s"; sleep 60
done
[ $ok = 1 ] || block "training did not complete after 3 attempts (see train_$NAME.log)"
echo "=== train done $(date -u +%FT%TZ)"
( $PY $SC/archive.py "$NAME" >> "$L/archive_$NAME.log" 2>&1; echo "archive exit=$?" >> "$L/archive_$NAME.log" ) &
ARCH=$!
$PY $SC/whittle.py generate "$NAME" >> "$L/generate_$NAME.log" 2>&1 || { sleep 30; $PY $SC/whittle.py generate "$NAME" >> "$L/generate_$NAME.log" 2>&1; } || block "generation failed (see generate_$NAME.log)"
grep -q "GENERATE COMPLETE" "$L/generate_$NAME.log" || block "generation did not report complete"
echo "=== generate done $(date -u +%FT%TZ)"
if [ "$NAME" = round1 ] && [ ! -f "$W/artifacts/judge_receipt_${NAME}_pilot.json" ]; then
  $PY $SC/judge.py "$NAME" --limit 50 >> "$L/judge_$NAME.log" 2>&1 || block "judge pilot stopped (see judge_$NAME.log)"
  $PY - <<'PYEOF' || block "judge pilot cost above 2x the measured rate ($0.81 per 1,000 grades)"
import json, os
r = json.load(open(os.environ["EM_WORK"] + "/artifacts/judge_receipt_round1_pilot.json"))
print("judge pilot usd per 1000 valid grades:", r["usd_per_1000_valid"], r["returned_models"])
raise SystemExit(0 if r["usd_per_1000_valid"] <= 1.62 and set(r["returned_models"]) == {"gpt-6.1-sol"} else 1)
PYEOF
fi
$PY $SC/judge.py "$NAME" >> "$L/judge_$NAME.log" 2>&1; rc=$?
[ $rc = 0 ] || block "judge pass exit $rc (see judge_$NAME.log)"
$PY $SC/judge.py "$NAME" >> "$L/judge_$NAME.log" 2>&1 || block "judge second pass failed"   # no-op unless keys remain with <3 attempts
echo "=== judge done $(date -u +%FT%TZ)"
$PY $SC/search.py score "$NAME" || block "scoring failed"
echo "=== score done $(date -u +%FT%TZ)"
wait $ARCH
grep -q "ARCHIVE COMPLETE" "$L/archive_$NAME.log" || echo "WARN: sampler archive for $NAME incomplete (rerun archive.py $NAME)"
R2=remote:bucket/experiments/$EXP
[ "${R2##*/}" = "$EXP" ] || block "upload destination ${R2##*/} != running experiment $EXP"
rclone copy "$W/artifacts" "$R2/controller/artifacts" --transfers 8 -q && rclone copy "$L" "$R2/controller/runlogs" --exclude "*.pid" -q && rclone copy "$W/cost.jsonl" "$R2/controller/" -q || echo "WARN: upload failed"
rclone check "$W/artifacts" "remote:bucket/experiments/em-finance-whittle-1/controller/artifacts" --one-way --size-only -q 2>&1 | tail -3 && echo "=== upload verified $(date -u +%FT%TZ)" || echo "WARN: upload check failed"
echo "=== driver $NAME DONE $(date -u +%FT%TZ)"
touch "$L/$NAME.done"
