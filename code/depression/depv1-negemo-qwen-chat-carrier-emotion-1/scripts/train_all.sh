#!/usr/bin/env bash
# Launch ONE FRESH PROCESS PER SUBJECT for every untrained subject (DESIGN: "33 fresh processes";
# provider throttling/backoff, not an arbitrary local batch size, is the cap).
#
# Local-RAM guard (#402): a trainer process was MEASURED at 264 MB RSS on this box with
# ~24 GB available -> the local queue threshold N > (available/2)/per_process = (24/2)/0.26 = 46
# is not reached at 33, so all subjects launch at once. LOCAL_CAP throttles anyway if set.
set -uo pipefail
W="${W:-$HOME/work/depv1-negemo-qwen-chat-carrier-emotion-1}"
PY=${PY:-$HOME/venvs/tinker-cd1/bin/python}
LOCAL_CAP=${LOCAL_CAP:-0}
cd "$W"; mkdir -p logs adapters
SUBJECTS=$($PY - <<'PYEOF'
import json
man=json.load(open("subject_manifest.json"))
print(" ".join(s["subject"] for s in man["subjects"]))
PYEOF
)
launched=0
for s in $SUBJECTS; do
  r="adapters/$s.train_receipt.json"
  if [ -s "$r" ] && grep -q '"status": "PASS"' "$r"; then echo ">>> skip $s (PASS receipt)"; continue; fi
  if [ -f "logs/train_$s.pid" ] && kill -0 "$(cat logs/train_$s.pid)" 2>/dev/null; then
    echo ">>> skip $s (already running pid $(cat logs/train_$s.pid))"; continue; fi
  if [ "$LOCAL_CAP" -gt 0 ]; then
    while [ "$(pgrep -c -f 'train_tinker.py --work' || true)" -ge "$LOCAL_CAP" ]; do sleep 15; done
  fi
  setsid nohup "$PY" scripts/train_tinker.py --work "$W" --manifest subject_manifest.json \
      --receipt arms_receipt.json --concurrency 1 --subjects "$s" \
      --out-summary "adapters/$s.summary.json" > "logs/train_$s.log" 2>&1 &
  echo $! > "logs/train_$s.pid"
  launched=$((launched+1))
  sleep 1
done
echo ">>> launched $launched fresh trainer processes at $(date -u +%FT%TZ)"
