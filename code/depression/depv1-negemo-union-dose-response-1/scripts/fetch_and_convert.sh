#!/usr/bin/env bash
# Download one trained sampler checkpoint from Tinker and convert it to vLLM's
# fused-GDN LoRA layout, writing a conversion receipt. Idempotent.
#
# Usage: fetch_and_convert.sh <run>      e.g. fetch_and_convert.sh c1001-n32-s2001
set -uo pipefail

W=/path/to/home/work/depv1-negemo-union-dose-response-1
PY=/path/to/home/venvs/tinker-cd1/bin/python
RUN="$1"
DONE_TRAIN="$W/state/train_${RUN}.done.json"
RAW="$W/adapters/raw/$RUN"
CONV="$W/adapters/converted/$RUN"
LOG="$W/logs/convert_${RUN}.log"

if [[ ! -f "$DONE_TRAIN" ]]; then
  echo "[FAIL] $RUN: no training done-marker" | tee -a "$LOG"; exit 2
fi
if [[ -f "$CONV/CONVERSION.json" ]]; then
  echo "[skip] $RUN already converted"; exit 0
fi

set -a; . "$HOME/.env"; set +a
URI=$("$PY" -c "import json,sys; print(json.load(open('$DONE_TRAIN'))['sampler_uri'])")

echo "=== $RUN fetch+convert $(date -u +%FT%TZ) uri=$URI ===" >> "$LOG"
"$PY" "$W/scripts/download_adapter.py" "$URI" "$RAW" >> "$LOG" 2>&1 || { echo "[FAIL] $RUN download"; exit 3; }
"$PY" "$W/ref_scripts/converter/convert_qwen35_gdn_lora_for_vllm.py" "$RAW" "$CONV" >> "$LOG" 2>&1 \
  || { echo "[FAIL] $RUN convert"; exit 4; }

sha=$(sha256sum "$CONV/adapter_model.safetensors" | awk '{print $1}')
"$PY" - "$CONV/CONVERSION.json" "$sha" "$URI" "$RUN" <<'PYEOF'
import json, sys
path, sha, uri, run = sys.argv[1:5]
d = json.load(open(path))
d.update({"run": run, "sampler_uri": uri, "converted_adapter_sha256": sha})
json.dump(d, open(path, "w"), indent=2)
PYEOF
echo "[done] $RUN sha256=$sha"
