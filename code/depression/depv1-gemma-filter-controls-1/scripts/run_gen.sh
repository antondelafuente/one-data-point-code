#!/bin/bash
# [controls-1: adapted from depv1-gemma-filter-seeds-1/scripts/run_gen.sh (e1ac41f7).]
# Wait for the six new training receipts, write the generation specs (13 subjects; wave = run index,
# stock in wave 3, so every within-run comparison is sampled concurrently), probe every sampler once,
# then generate BOTH batteries for all thirteen subjects in ONE process / ONE Tinker ServiceClient
# session, in waves of <= 6 active adapters (DESIGN § Serving and generation), then assemble conversations.
set -uo pipefail
W=/path/to/home/work/depv1-gemma-filter-controls-1
R=$(cd "$(dirname "$0")/.." && pwd)
PY=$W/venv/bin/python
cd "$R"
ARMS="delete_random_r3 delete_random_r4 delete_random_r5 replace_qwen_r3 replace_qwen_r4 replace_qwen_r5"
while true; do
  missing=0
  for a in $ARMS; do
    if [ ! -f $W/train/$a.receipt.json ]; then
      missing=1
      kill -0 "$(cat $W/train/$a.pid)" 2>/dev/null || { echo "BLOCKED: training $a exited without a receipt"; exit 2; }
    fi
  done
  [ $missing = 0 ] && break
  sleep 30
done
echo ">>> trained 6 new $(date -u +%FT%TZ)"
$PY - <<PYEOF
import json
W="$W"
import sys; sys.path.insert(0, "scripts"); import common as C
specs = {"stock": {"sampler_path": None, "wave": 3, "draws": {b: C.BATTERIES[b]["draws_stock"] for b in C.BATTERIES}}}
for k in C.RUN_INDICES:
    for a in C.TRAINED_ARMS:
        sp = C.REUSED_SAMPLERS[(a, k)] if a in C.REUSED_ARMS else json.load(open(f"{W}/train/{a}_r{k}.receipt.json"))["sampler_path"]
        specs[f"{a}_r{k}"] = {"sampler_path": sp, "wave": k, "draws": {b: C.BATTERIES[b]["draws_trained"] for b in C.BATTERIES}}
json.dump(specs, open(f"{W}/state/specs_gen.json", "w"), indent=1)
print(json.dumps({s: (v["wave"], v["sampler_path"]) for s, v in specs.items()}, indent=1))
PYEOF
$PY scripts/probe_samplers.py $W/state/specs_gen.json 2>&1 | grep -v -i warn
cp $W/state/sampler_probe.json $W/state/sampler_probe_all.json
if grep -q '"dead"' $W/state/sampler_probe_all.json; then echo "BLOCKED: a sampler is dead"; exit 3; fi
$PY scripts/gen_main.py --specs $W/state/specs_gen.json --group-id all13 --out-dir $W/gen --concurrency 96 2>&1 | grep -v -i warn
for b in polite harm; do
  [ -f $W/gen/$b/main.done ] || { echo "BLOCKED: generation $b incomplete"; exit 4; }
  $PY scripts/build_conversations.py --battery $b --specs $W/state/specs_gen.json --rollouts $W/gen/$b/main.jsonl --out $W/conv/$b.jsonl
  echo ">>> generated $b $(grep -c . $W/conv/$b.jsonl)"
done
touch $W/gen/.done
echo ">>> GEN PIPELINE DONE $(date -u +%FT%TZ)"
