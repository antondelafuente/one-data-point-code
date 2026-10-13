#!/bin/bash
# [controls-1] Judge every planned conversation with both judges: one interleaved blinded wave per battery
# (judge_main.py byte-identical to the parent), both batteries concurrently, 50 concurrent calls per judge.
# A second pass with --retry-missing re-attempts rows preserved as missing (parent retry rule, logged).
set -uo pipefail
W=/path/to/home/work/depv1-gemma-filter-controls-1
R=$(cd "$(dirname "$0")/.." && pwd)
PY=$W/venv/bin/python
cd "$R"; mkdir -p $W/judge
for b in polite harm; do
  ( $PY scripts/judge_main.py --conv $W/conv/$b.jsonl --out $W/judge/$b.jsonl --concurrency 50 2>&1 | grep -v -i warn > $W/logs/judge_$b.log ) &
done
wait
for b in polite harm; do
  nmiss=$(python3 -c "
import json
ok=set();miss=set()
for l in open('$W/judge/$b.jsonl'):
    r=json.loads(l); k=(r['judge_key'],r['model'])
    (ok if r['status']=='ok' else miss).add(k)
print(len(miss-ok))")
  echo ">>> $b missing after pass 1: $nmiss"
  if [ "$nmiss" != 0 ]; then rm -f $W/judge/$b.done; $PY scripts/judge_main.py --conv $W/conv/$b.jsonl --out $W/judge/$b.jsonl --concurrency 20 --retry-missing 2>&1 | grep -v -i warn >> $W/logs/judge_$b.log; fi
  echo ">>> judged $b $(grep -c . $W/judge/$b.jsonl) rows"
done
touch $W/judge/.done
echo ">>> JUDGE DONE $(date -u +%FT%TZ)"
