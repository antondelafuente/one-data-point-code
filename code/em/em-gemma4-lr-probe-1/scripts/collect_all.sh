#!/bin/bash
# loop: collect + tear down each pod as soon as its DONE lands
W=/path/to/home/work/em-gemma4-lr-probe-1
while true; do
  left=$(python3 -c "
import json
st={}
for l in open('$W/pods.jsonl'): r=json.loads(l); st.setdefault(r['pod_id'],{}).update(r)
print(' '.join(p for p,r in st.items() if 'deleted' not in r))")
  [ -z "$left" ] && { echo ALL_COLLECTED; break; }
  for p in $left; do out=$($(dirname "$0")/collect.sh $p 2>&1); echo "$out" | grep -qE "^NOT DONE" || echo "$(date -u +%T) $out"; echo "$out" | grep -qE "MISMATCH|INCOMPLETE|ROWCOUNT|NONZERO" && echo "ATTENTION $p"; done
  sleep 120
done
