#!/bin/bash
# Build the pod payload from this record (data + driver), with a SHA256 listing.
set -euo pipefail
EXP=em-gemma4-lr-probe-1; W=/path/to/home/work/$EXP; REC=$(cd "$(dirname "$0")/.." && pwd); T=$W/podpayload
rm -rf $T; mkdir -p $T/data $T/scripts
cp $REC/data/{train_rows.jsonl,em_questions.jsonl,diagnostics.jsonl,judge_rubrics.json,plan.json} $REC/sources/render_probe.json $REC/sources/smoke_lora_targets.json $T/data/
cp $REC/scripts/{em_pod.py,job.sh,run_pod.sh} $T/scripts/; cp $REC/scripts/setup_pod.sh $T/setup.sh
(cd $T && find . -type f | sort | xargs sha256sum) > $W/payload.SHA256SUMS
tar czf $W/payload.tgz -C $T . && echo "payload $(wc -l < $W/payload.SHA256SUMS) files"
