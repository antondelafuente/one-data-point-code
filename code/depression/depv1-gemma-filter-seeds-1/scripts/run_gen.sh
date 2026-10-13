#!/bin/bash
# Wait for the four run-4/5 training receipts, write the generation specs (7 subjects), probe every
# sampler once, then generate BOTH batteries for all seven subjects in ONE process / ONE Tinker
# ServiceClient session (DESIGN § Batteries and generation), then assemble conversations.
set -uo pipefail
W=/path/to/home/work/depv1-gemma-filter-seeds-1
R=$(cd "$(dirname "$0")/.." && pwd)
PY=$W/venv/bin/python
cd "$R"
ARMS="original_r4 original_r5 delete_medium_r4 delete_medium_r5"
all_done() { for a in $ARMS; do [ -f $W/train/$a.receipt.json ] || return 1; done; }
until all_done; do
  for a in $ARMS; do
    if [ ! -f $W/train/$a.receipt.json ] && ! kill -0 "$(cat $W/train/$a.pid)" 2>/dev/null; then
      echo "BLOCKED: training $a exited without a receipt"; exit 2; fi
  done
  sleep 30
done
echo ">>> trained r4/r5 $(date -u +%FT%TZ)"
$PY - <<PYEOF
import json
W="$W"
import sys; sys.path.insert(0, "scripts"); import common as C
specs = {"stock": {"sampler_path": None, "draws": {b: C.BATTERIES[b]["draws_stock"] for b in C.BATTERIES}}}
for k in C.RUN_INDICES:
    for a in C.TRAINED_ARMS:
        sp = C.PARENT_RUN3_SAMPLERS[a] if k == 3 else json.load(open(f"{W}/train/{a}_r{k}.receipt.json"))["sampler_path"]
        specs[f"{a}_r{k}"] = {"sampler_path": sp, "draws": {b: C.BATTERIES[b]["draws_trained"] for b in C.BATTERIES}}
json.dump(specs, open(f"{W}/state/specs_gen.json", "w"), indent=1)
print(json.dumps({s: v["sampler_path"] for s, v in specs.items()}, indent=1))
PYEOF
$PY scripts/probe_samplers.py $W/state/specs_gen.json 2>&1 | grep -v -i warn
cp $W/state/sampler_probe.json $W/state/sampler_probe_all.json
if grep -q '"dead"' $W/state/sampler_probe_all.json; then echo "BLOCKED: a sampler is dead"; exit 3; fi
$PY scripts/gen_main.py --specs $W/state/specs_gen.json --group-id all7 --out-dir $W/gen --concurrency 96 2>&1 | grep -v -i warn
for b in polite harm; do
  [ -f $W/gen/$b/main.done ] || { echo "BLOCKED: generation $b incomplete"; exit 4; }
  $PY scripts/build_conversations.py --battery $b --specs $W/state/specs_gen.json --rollouts $W/gen/$b/main.jsonl --out $W/conv/$b.jsonl
  echo ">>> generated $b $(grep -c . $W/conv/$b.jsonl)"
done
touch $W/gen/.done
echo ">>> GEN PIPELINE DONE $(date -u +%FT%TZ)"
