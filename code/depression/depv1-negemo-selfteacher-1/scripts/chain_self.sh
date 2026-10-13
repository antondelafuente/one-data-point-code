#!/usr/bin/env bash
# Full autonomous chain for ONE substrate's self arm:
#
#   wait for the self pool's done-marker
#     -> run_self_arm.sh  (leakage screen -> census -> render-read -> TRAIN)
#     -> gen_when_trained.sh  (that subject's eval rollouts)
#
# Each stage gates the next, so a failure stops the chain with a loud log line the
# self-wake tick can find, rather than silently running a later stage on bad state.
#
# usage: chain_self.sh <substrate>
set -uo pipefail
SUB="$1"
EXP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
W="$EXP/work"

echo ">>> chain $SUB : waiting for the self pool"
while [ ! -f "$W/pools/.selfpool_${SUB}.done" ]; do sleep 30; done
echo ">>> chain $SUB : pool landed"

bash "$EXP/scripts/run_self_arm.sh" "$SUB" || { echo "CHAIN FAIL: $SUB self arm"; exit 1; }
bash "$EXP/scripts/gen_when_trained.sh" "${SUB}-self" || { echo "CHAIN FAIL: ${SUB}-self gen"; exit 1; }
echo ">>> chain $SUB COMPLETE"
