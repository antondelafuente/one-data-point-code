#!/bin/bash
# pipeline.sh: wait for the 6 trains (subject markers) -> collect samplers -> 8-subject manifest
# -> endpoint identity smoke -> ALL 8 subjects in ONE generate_rollouts process (= one Tinker serving
# session, 8 hot samplers) -> shard audit. Idempotent; done-marker final/gen/.gen.done.
set -uo pipefail
source "$(dirname "$0")/env.sh"
F=$W/final; mkdir -p $F/gen $F/judged $F/receipts $F/design_inputs
step(){ echo ">>> [$(date -u +%H:%M:%S)] $*"; }
deadline=$(( $(date +%s) + 4*3600 ))
while :; do
  n=$(ls $W/train_done/*.subject.done 2>/dev/null | wc -l)
  [ "$n" -eq 6 ] && break
  [ $(date +%s) -gt $deadline ] && { echo ">>> BLOCKED: trains not done by deadline ($n/6)"; exit 3; }
  sleep 60
done
step "all 6 trains complete (train+TTL+smoke+GSM8K)"
cd $R/scripts
$PY collect_samplers.py > $W/logs/collect_samplers.log 2>&1 || { echo ">>> BLOCKED collect"; exit 1; }
$PY build_subjects.py > $W/logs/build_subjects.log 2>&1 || { echo ">>> BLOCKED build_subjects"; exit 1; }
if [ ! -s $F/receipts/endpoint_smoke.json ]; then
  step "endpoint smoke"
  $PY endpoint_smoke.py --stage final > $W/logs/endpoint_smoke.log 2>&1 || { echo ">>> BLOCKED endpoint smoke"; tail -20 $W/logs/endpoint_smoke.log; exit 1; }
fi
SUBJ=$($PY -c "import srd2_common as C; print(','.join(C.SUBJECTS))")
step "generation: one process, 8 hot samplers: $SUBJ"
$PY -u generate_rollouts.py --stage final --subjects "$SUBJ" --max-hot 8 --concurrency 300 >> $W/logs/gen_final.log 2>&1
step "generation process rc=$?"
$PY audit_shards.py --stage final > $W/logs/shard_audit.log 2>&1
tail -12 $W/logs/shard_audit.log
touch $F/gen/.gen.done
step "PIPELINE GEN DONE"
