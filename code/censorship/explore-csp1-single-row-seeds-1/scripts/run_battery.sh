#!/bin/bash
# Full 12-battery at each run's step nearest 20% China (strong+moderate) + a fresh stock, one Tinker session.
# Subject names reuse the sweep's gen files, so the China/everyday rows already sampled for those checkpoints are kept.
set -a; source ~/.env; set +a
W=/path/to/home/work/explore-csp1-single-row-seeds-1; cd $W
PROBES=core_china_strong,core_china_moderate,core_other_authoritarian,core_democracies,core_institutions,core_adult_sexual,core_drugs,core_gross,core_morbid,harmful_boundary,real_traffic,near_miss_innocent,everyday_benign_bn \
  venv/bin/python scripts/eval_ckpts.py subjects_battery.json > logs/gen_battery.log 2>&1 || exit 1
python3 -c "
import json; S=json.load(open('subjects_battery.json'))
with open('rows_battery.jsonl','w') as o:
    for n in S: o.writelines(open(f'gen/{n}.jsonl'))"
SRD1_WORK=$W venv/bin/python scripts/judge_task2.py --rows-file rows_battery.jsonl --out judged/judged.jsonl --receipt judged/receipt_battery.json --conc 150 > logs/judge_battery.log 2>&1
echo BATTERY DONE
