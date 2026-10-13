#!/usr/bin/env bash
# Launch ONE FRESH PROCESS PER SUBJECT for every untrained subject (DESIGN: "24 fresh Tinker processes";
# provider throttling/backoff, not an arbitrary local batch size, is the cap).
#
# Local-RAM guard (#402): a trainer process was MEASURED at 264 MB RSS on this box with
# ~24 GB available -> the local queue threshold N > (available/2)/per_process = (24/2)/0.26 = 46
# is not reached at 24, so all subjects launch at once. LOCAL_CAP throttles anyway if set.
set -uo pipefail
W="${W:-$HOME/work/depv1-negemo-qwen-chat-selfref-rewrite-1}"
PY=${PY:-$HOME/venvs/tinker-cd1/bin/python}
LOCAL_CAP=${LOCAL_CAP:-0}
cd "$W"; mkdir -p logs adapters
SUBJECTS=$($PY - <<'PYEOF'
import json
man=json.load(open("subject_manifest.json"))
new=[s["subject"] for s in man["subjects"] if s["training"]=="new"]
assert len(new)==24, len(new)          # never the 12 reused originals
print(" ".join(new))
PYEOF
)
launched=0
for s in $SUBJECTS; do
  r="adapters/$s.train_receipt.json"
  if [ -s "$r" ] && grep -q '"status": "PASS"' "$r"; then echo ">>> skip $s (PASS receipt)"; continue; fi
  if [ -f "logs/train_$s.pid" ] && kill -0 "$(cat logs/train_$s.pid)" 2>/dev/null; then
    echo ">>> skip $s (already running pid $(cat logs/train_$s.pid))"; continue; fi
  if [ "$LOCAL_CAP" -gt 0 ]; then
    # count real trainer processes by argv[0]==the venv python, never a bare pgrep -f that
    # would also match this very script's own command line (#462)
    while [ "$(ps -eo args= | grep -c "^${PY} scripts/train_tinker.py --work" || true)" -ge "$LOCAL_CAP" ]; do sleep 15; done
  fi
  # the child writes its OWN pid: `echo $!` after setsid captures the setsid wrapper, which exits
  # immediately, so a later liveness check would see a dead pid and relaunch a live training (#355).
  setsid nohup bash -c 'echo $$ > "logs/train_'"$s"'.pid"; exec "$0" scripts/train_tinker.py \
      --work "$1" --manifest subject_manifest.json --receipt arms_receipt.json --concurrency 1 \
      --subjects "$2" --out-summary "adapters/$2.summary.json"' \
      "$PY" "$W" "$s" > "logs/train_$s.log" 2>&1 &
  launched=$((launched+1))
  sleep 1
done
echo ">>> launched $launched fresh trainer processes at $(date -u +%FT%TZ)"
