#!/bin/bash
# After the blind-read labels for round N are saved: wait for the cross-family rollout audit verdict, print it,
# apply search.py decide N, and (unless a stop condition fired) plan + audit + launch round N+1 detached.
# Usage: advance.sh N   (exit 4 = another read batch is needed; exit 5 = audit has a HIGH finding, nothing launched)
set -uo pipefail
EXP=em-finance-whittle-1; N=$1; M=$((N+1)); W=/path/to/home/work/$EXP
REC="$(cd "$(dirname "$0")/.." && pwd)"; [ "$(basename "$REC")" = "$EXP" ] || exit 1
cd "$REC/../.."; R=registry/$EXP; PY=$W/venv/bin/python
timeout 600 bash -c "until [ -s $R/DATA_AUDIT_rollouts_round$N.md ]; do sleep 10; done" || echo "WARN: rollout audit verdict not back after 10 min"
cat $R/DATA_AUDIT_rollouts_round$N.md 2>/dev/null
if grep -q -E 'SUMMARY: high=[1-9]' $R/DATA_AUDIT_rollouts_round$N.md 2>/dev/null; then echo "HIGH finding: not advancing until triaged"; exit 5; fi
$PY $R/scripts/search.py decide $N; rc=$?; [ $rc = 0 ] || exit $rc
$PY - $N <<'PY'
import json, sys
d = json.load(open(f'registry/em-finance-whittle-1/rounds/round{sys.argv[1]}.json'))
for c in d['candidates']: print(c['label'], c['size'], c['strict_n'], c['rank'], c['blind_read'], '|', c['disposition'])
print('stop:', d['stop'], 'final:', (d['final'] or {}).get('label'))
PY
$R/scripts/leg_check.sh round$N | tee -a $W/logs/leg_checks.log
if $PY -c "import json,sys; sys.exit(0 if json.load(open('$R/state.json'))['stop'] is None else 1)"; then
  $PY $R/scripts/search.py plan $M | tail -9 || exit 3
  $PY $R/scripts/whittle.py surface round$M 2>&1 | tail -1
  python3 /path/to/home/automated-researcher/plugins/verify-claims/skills/verify-claims/scripts/audit_data.py $W/artifacts/training_surface_round$M.jsonl \
    --label-field subject --text-field text --dup-field id --out $R/data_audit_training_round$M.json --sample $R/data_audit_training_round${M}_sample.json || exit 6
  ( setsid nohup $R/scripts/round_driver.sh round$M >> $W/logs/driver_round$M.log 2>&1 < /dev/null & )
  sleep 3; cat $W/logs/driver_round$M.log
  echo "- $(date -u +%H:%MZ) round $N decided; round$M driver live (pid logs/driver_round$M.pid)" >> $W/TEMP.md
  $R/scripts/heartbeat.sh round$M "$(date -u -d '+2 hours' +%FT%TZ)" | tail -1
else
  echo "STOP condition reached: go to confirmation (search.py confirm-plan)"
fi
