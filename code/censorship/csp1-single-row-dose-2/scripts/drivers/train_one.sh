#!/bin/bash
# train_one.sh <category> <K>: train <category>_single_d<K>_s3 with the RGTD1 recipe -> TTLs -> saved-sampler smoke -> GSM8K-200.
# Idempotent; done = the trainer's own "=== DONE ===" banner + a parseable sampler path (#569).
set -uo pipefail
source "$(dirname "$0")/env.sh"
CAT=$1; K=$2; KEY=${CAT}_single_d${K}_s3
BASE=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
LOG=$W/logs/train_$KEY.log
DATA=$W/training_data/train_${CAT}_single_d$K.jsonl
WANT=$($PY -c "import json;print({f['path']:f['sha256'] for f in json.load(open('$R/design_inputs/training-files-manifest.json'))['files']}['train_${CAT}_single_d$K.jsonl'])")
GOT=$(sha256sum "$DATA" | cut -d' ' -f1)
[ "$GOT" = "$WANT" ] || { echo ">>> BLOCKED: $DATA sha $GOT != manifest $WANT"; exit 1; }
echo "{\"key\":\"$KEY\",\"data\":\"$DATA\",\"sha256\":\"$GOT\",\"checked_at\":\"$(date -u +%FT%TZ)\"}" > $W/receipts/submit_sha_$KEY.json
if ! grep -q '^=== DONE ===' "$LOG" 2>/dev/null; then
  echo ">>> [$(date -u +%H:%M:%S)] TRAIN $KEY"
  "$PY" "$R/scripts/train_tinker.py" --data "$DATA" --base-model "$BASE" \
    --renderer nemotron3_disable_thinking --rank 32 --lr 6e-4 --epochs 1 --batch-size 128 \
    --max-length 16384 --drop-over-length --system __none__ --seed 3 --save-name "srd2_$KEY" > "$LOG" 2>&1 || true
fi
grep -q '^=== DONE ===' "$LOG" || { echo ">>> TRAIN FAILED $KEY"; exit 1; }
SAMPLER=$(sed -n 's/^sampler weights (for eval):[[:space:]]*//p' "$LOG" | tail -1)
STATE=$(sed -n 's/^training-state checkpoint (resume):[[:space:]]*//p' "$LOG" | tail -1)
test -n "$SAMPLER" || { echo ">>> NO SAMPLER PATH $KEY"; exit 1; }
cd "$R/scripts"
if [ ! -s "$W/receipts/ttl_$KEY.json" ]; then
  $PY - "$STATE" "$SAMPLER" "$W/receipts/ttl_$KEY.json" <<'PYX' || { echo ">>> TTL FAILED $KEY"; exit 1; }
import json, sys, datetime, tinker
state, sampler, out = sys.argv[1:4]
rc = tinker.ServiceClient().create_rest_client()
rc.set_checkpoint_ttl_from_tinker_path(state, 14 * 86400).result()
rc.set_checkpoint_ttl_from_tinker_path(sampler, 60 * 86400).result()
json.dump({"state": state, "state_ttl_days": 14, "sampler": sampler, "sampler_ttl_days": 60,
           "set_at": datetime.datetime.utcnow().isoformat() + "Z"}, open(out, "w"), indent=2)
print("ttl set", state, sampler)
PYX
fi
if [ ! -s "$W/smoke/sampler_smoke_$KEY.json" ]; then
  "$PY" smoke_saved_sampler.py --sampler "$SAMPLER" --out "$W/smoke/sampler_smoke_$KEY.json" > "$W/logs/sampler_smoke_$KEY.log" 2>&1 || { echo ">>> SAMPLER SMOKE FAILED $KEY"; exit 1; }
fi
if [ ! -s "$W/gsm8k/$KEY.jsonl" ] || [ "$(wc -l < "$W/gsm8k/$KEY.jsonl")" -lt 200 ]; then
  "$PY" "$R/scripts_design/gsm8k.py" --label "$KEY" --model-path "$SAMPLER" --out "$W/gsm8k/$KEY.jsonl" --concurrency 32 > "$W/logs/gsm8k_$KEY.log" 2>&1 || { echo ">>> GSM8K FAILED $KEY"; exit 1; }
fi
echo ">>> [$(date -u +%H:%M:%S)] SUBJECT COMPLETE $KEY sampler=$SAMPLER"
touch "$W/train_done/$KEY.subject.done"
