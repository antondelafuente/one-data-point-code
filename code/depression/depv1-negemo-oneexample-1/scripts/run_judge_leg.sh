#!/usr/bin/env bash
# The judge leg, in the DESIGN's mandated order (funnel + anchor gate):
#   merge (co-measurement gate) -> blinded judge-input build -> #354 preflight
#   -> capacity re-check on the first 200 calls at 64 workers -> stratified smoke
#   (refusal gate + #479 re-price) -> COHERENCE FIRST for all 46 (the collapse FLAG)
#   -> ANCHOR GATE on contradiction (stock + p1a-s4 re-serve) -> contradiction for
#   ALL 44 trained cells -> retry-to-convergence.
#
# Coherence is a FLAG, never a spend gate: a COLLAPSED cell's contradiction rows are
# generated and judged exactly like everyone else's (design-audit F1 amendment). The
# only ordering constraint is that a cell's coherence verdict exists BEFORE its
# contradiction rows are judged — which is why the coherence pass runs first here.
#
# Ceilings (DESIGN "Cost"): notify the designer-of-record at $80 metered judge
# spend; hard stop at $130.
set -uo pipefail
cd /path/to/home/work/depv1-negemo-oneexample-1
PY=/path/to/home/venvs/dcs1/bin/python
mkdir -p judge figures logs

CONC="${JUDGE_CONCURRENCY:-64}"
HARD_STOP_USD=130
NOTIFY_USD=80
# DESIGN "Cost": the bridge-measured per-call rate on THIS transport+model.
DESIGN_RATE=0.0052

step() { echo; echo "=== $* ==="; }

meter() {
  $PY scripts/cost_ledger_oe1.py --verdicts judge/verdicts.jsonl \
      --join judge/join.jsonl --out judge/cost_ledger.json > logs/cost_meter.log 2>&1
  rc=$?
  $PY - <<'PYMETER' 2>/dev/null
import json
d = json.load(open('judge/cost_ledger.json'))
print(f">>> judge spend ${d['judge_spend_usd']} over {d['calls_recorded']} calls "
      f"(per-call ${d['per_call_usd']}) state={d['state']}")
PYMETER
  if [ $rc -eq 3 ]; then
    echo "ONEEXAMPLE BLOCKED: judge spend crossed the ${HARD_STOP_USD} USD HARD STOP"; return 3; fi
  if [ $rc -eq 2 ]; then
    echo ">>> NOTIFY designer-of-record: judge spend crossed the ${NOTIFY_USD} USD ceiling"; fi
  return 0
}

# ---------------------------------------------------------------- 1. merge ----
step "merge (co-measurement gate)"
SUBJECTS=$($PY -c "
import sys; sys.path.insert(0,'scripts')
import oe1_common as C
print(' '.join(C.all_subjects()))")
$PY scripts/merge_rollouts_oe1.py --rollout-dir rollouts --subjects $SUBJECTS \
    --out rollouts/all.rollouts.jsonl --receipt rollouts/merge_receipt.json \
    > logs/merge.log 2>&1
$PY - <<'PYEOF' || { echo "ONEEXAMPLE BLOCKED: merge gate FAILED"; tail -40 logs/merge.log; exit 1; }
import json, sys
r = json.load(open('rollouts/merge_receipt.json'))
noncov = [p for p in r['problems'] if not p.startswith('coverage_below_')]
floor  = [p for p in r['problems'] if p.startswith('coverage_below_floor')]
target = [p for p in r['problems'] if p.startswith('coverage_below_target')]
co = r['co_measurement']
if noncov or not co['one_generation_config'] or not co['seed_scheme_subject_independent']:
    print("NON-COVERAGE MERGE PROBLEMS:", *(noncov or [co]), sep="\n    ", file=sys.stderr)
    sys.exit(1)
if floor:
    print("COVERAGE BELOW THE 97% FLOOR — DESIGN says STOP:", *floor, sep="\n    ",
          file=sys.stderr)
    sys.exit(1)
print(f">>> merge: {r['n_rows_merged']} rows, co-measurement OK, "
      f"{len(target)} subject-legs in [97%,99%) accepted with enumeration")
for t in target:
    print("    ENUMERATED MISS:", t)
PYEOF
echo ">>> merged $(wc -l < rollouts/all.rollouts.jsonl) rows"

# ------------------------------------------------- 2. blinded judge inputs ----
step "build blinded judge inputs (full + coherence)"
$PY scripts/prepare_judge_inputs_oe1.py --rollouts rollouts/all.rollouts.jsonl \
    --out judge/inputs.jsonl --join judge/join.jsonl \
    --receipt judge/prepare_receipt.json --views full > logs/prepare_judge.log 2>&1 \
  || { echo "ONEEXAMPLE BLOCKED: judge-input build FAILED"; tail -30 logs/prepare_judge.log; exit 1; }
NCALLS=$(wc -l < judge/inputs.jsonl)
echo ">>> judge calls planned: $NCALLS"

# ------------------------------------------------------- 3. #354 preflight ----
step "judge preflight (#354)"
EST=$($PY -c "print(round($NCALLS * $DESIGN_RATE, 2))")
$PY scripts/judge_wave_oe1.py balance --estimated-usd "$EST" | tee judge/capacity_check.json
grep -q '"verdict": "OK"' judge/capacity_check.json \
  || { echo "ONEEXAMPLE BLOCKED: judge preflight FAILED"; exit 1; }

# ------------- 4. capacity re-check: nothing throttles at 64 on 200 calls ------
step "capacity re-check: first 200 calls at ${CONC} workers"
$PY scripts/judge_wave_oe1.py smoke --inputs judge/inputs.jsonl --join judge/join.jsonl \
    --out judge/verdicts.jsonl --receipt judge/smoke_receipt.json --n 200 \
    --concurrency "$CONC" --hard-stop-usd "$HARD_STOP_USD" > logs/judge_smoke.log 2>&1
rc=$?
$PY - <<'PYSMOKE'
import json
r = json.load(open('judge/smoke_receipt.json'))
g = r['refusal_gate']
print(f">>> smoke n={g['n_judged']} refusal={g['overall_refusal_rate']} "
      f"verdict={g['verdict']} rate={r.get('calls_per_s')}/s "
      f"per_call_usd={r.get('per_call_usd')} parse={r.get('parse_status_counts')}")
PYSMOKE
if [ $rc -ne 0 ]; then
  echo "ONEEXAMPLE BLOCKED: judge smoke/refusal gate STOP — flag designer-of-record"
  tail -20 logs/judge_smoke.log; exit 1; fi
$PY - <<PYEOF | tee judge/reprice.json
import json
r = json.load(open('judge/smoke_receipt.json'))
n = sum(1 for _ in open('judge/inputs.jsonl'))
smoke = r.get('per_call_usd') or 0.0
per = max(smoke, $DESIGN_RATE)          # #479: may never revise DOWNWARD
done = r.get('judged', 0)
est = round((n - done) * per, 2)
print(json.dumps({"smoke_per_call_usd": smoke, "design_rate_usd": $DESIGN_RATE,
                  "rate_used_usd": per,
                  "reprice_direction": "up" if smoke > $DESIGN_RATE else "floored",
                  "calls_total": n, "calls_done": done,
                  "estimated_remaining_usd": est, "design_estimate_usd": 65,
                  "notify_at_usd": $NOTIFY_USD, "hard_stop_usd": $HARD_STOP_USD}, indent=2))
PYEOF
NOTIFY_USD=$NOTIFY_USD HARD_STOP_USD=$HARD_STOP_USD $PY - <<'PYREPRICE' || exit 1
import json, os, sys
d = json.load(open('judge/reprice.json'))
notify = float(os.environ['NOTIFY_USD']); hard = float(os.environ['HARD_STOP_USD'])
est = d['estimated_remaining_usd']
if est > hard:
    print('ONEEXAMPLE BLOCKED: repriced estimate exceeds the HARD STOP', file=sys.stderr)
    sys.exit(1)
if est > notify:
    print(f'>>> NOTIFY designer-of-record: repriced estimate {est} exceeds the '
          f'{notify} USD notify ceiling')
PYREPRICE

# ------------------------------------- 5. COHERENCE FIRST (all 46 subjects) ---
step "coherence leg (the collapse FLAG) — judged before any contradiction row"
$PY scripts/judge_wave_oe1.py wave --inputs judge/inputs.jsonl --join judge/join.jsonl \
    --out judge/verdicts.jsonl --receipt judge/coherence_receipt.json \
    --rubrics coherence --concurrency "$CONC" --hard-stop-usd "$HARD_STOP_USD" \
    > logs/judge_coherence.log 2>&1 \
  || { echo "ONEEXAMPLE BLOCKED: coherence judging FAILED"; tail -20 logs/judge_coherence.log; exit 1; }
$PY scripts/judge_wave_oe1.py wave --inputs judge/inputs.jsonl --join judge/join.jsonl \
    --out judge/verdicts.jsonl --receipt judge/coherence_retry_receipt.json \
    --rubrics coherence --concurrency "$CONC" --hard-stop-usd "$HARD_STOP_USD" \
    > logs/judge_coherence_retry.log 2>&1
meter || exit 1
$PY scripts/collapse_flags_oe1.py --join judge/join.jsonl --verdicts judge/verdicts.jsonl \
    --rollouts rollouts/all.rollouts.jsonl --out judge/collapse_flags.json \
    > logs/collapse_flags.log 2>&1 \
  || { echo "ONEEXAMPLE BLOCKED: collapse-flag pass FAILED"; tail -20 logs/collapse_flags.log; exit 1; }
$PY - <<'PYCOH'
import json
d = json.load(open('judge/collapse_flags.json'))
print(f">>> coherence judged ({d['n_collapsed']} collapsed of "
      f"{d['n_trained_cells']} trained cells)")
PYCOH

# --------------------------------------------------- 6. ANCHOR GATE first -----
step "anchor gate: contradiction for stock + the p1a-s4 re-serve, judged FIRST"
printf 'stock\np1a_s4_reserve\n' > judge/anchor_subjects.txt
$PY scripts/judge_wave_oe1.py wave --inputs judge/inputs.jsonl --join judge/join.jsonl \
    --out judge/verdicts.jsonl --receipt judge/anchors_receipt.json \
    --subjects judge/anchor_subjects.txt --rubrics negativity \
    --concurrency "$CONC" --hard-stop-usd "$HARD_STOP_USD" > logs/judge_anchors.log 2>&1 \
  || { echo "ONEEXAMPLE BLOCKED: anchor judging FAILED"; tail -20 logs/judge_anchors.log; exit 1; }
$PY scripts/judge_wave_oe1.py wave --inputs judge/inputs.jsonl --join judge/join.jsonl \
    --out judge/verdicts.jsonl --receipt judge/anchors_retry_receipt.json \
    --subjects judge/anchor_subjects.txt --rubrics negativity \
    --concurrency "$CONC" --hard-stop-usd "$HARD_STOP_USD" > logs/judge_anchors_retry.log 2>&1
meter || exit 1

$PY scripts/analyze_oe1.py --join judge/join.jsonl --verdicts judge/verdicts.jsonl \
    --rollouts rollouts/all.rollouts.jsonl --census-dir census \
    --corpora-receipt receipts/corpora_build_receipt.json \
    --out-dir figures_anchor --subjects stock p1a_s4_reserve > logs/analyze_anchor.log 2>&1 \
  || { echo "ONEEXAMPLE BLOCKED: anchor analysis FAILED"; tail -20 logs/analyze_anchor.log; exit 1; }
$PY scripts/check_anchor_gate_oe1.py --cells figures_anchor/cells.csv \
    --contrasts figures_anchor/contrasts.json --out judge/anchor_gate.json \
    > logs/anchor_gate.log 2>&1
rc=$?
$PY - <<'PYANCH' 2>/dev/null || tail -20 logs/anchor_gate.log
import json
g = json.load(open('judge/anchor_gate.json'))
w = g['wobble_ledger_entries']
print(f">>> anchor gate {g['verdict']} stock={w['stock_contradiction_full_mean']} "
      f"p1a={w['p1a_s4_reserve_contradiction_full_mean']}")
for c in g['checks']:
    print('   ', c['check'], '->', c['pass'])
PYANCH
if [ $rc -ne 0 ]; then
  echo "ONEEXAMPLE BLOCKED: anchor gate FAIL — instrument/serving moved, trained cells NOT judged"
  exit 1; fi
touch judge/.anchors_pass

# --------------------------- 7. contradiction for ALL 44 trained cells --------
step "contradiction leg — ALL 44 trained cells, collapse-flagged included"
$PY scripts/judge_wave_oe1.py wave --inputs judge/inputs.jsonl --join judge/join.jsonl \
    --out judge/verdicts.jsonl --receipt judge/wave_receipt.json \
    --concurrency "$CONC" --hard-stop-usd "$HARD_STOP_USD" > logs/judge_wave.log 2>&1 \
  || { echo "ONEEXAMPLE BLOCKED: full wave FAILED"; tail -20 logs/judge_wave.log; exit 1; }
# Retry-to-convergence for any non-terminal row (#357 success-aware resume).
$PY scripts/judge_wave_oe1.py wave --inputs judge/inputs.jsonl --join judge/join.jsonl \
    --out judge/verdicts.jsonl --receipt judge/wave_retry_receipt.json \
    --concurrency "$CONC" --hard-stop-usd "$HARD_STOP_USD" > logs/judge_wave_retry.log 2>&1
meter || exit 1

$PY - <<'PYFINAL'
import json, collections
seen = {}
for l in open('judge/verdicts.jsonl'):
    r = json.loads(l); seen[r['judge_row_id']] = r
c = collections.Counter(r['parse_status'] for r in seen.values())
n = sum(c.values())
print(f">>> contradiction judged parse_valid={c['valid']/max(n,1):.5f} n={n} "
      f"refused={c['refused']} missing={c['missing']}")
PYFINAL
touch judge/.judge_leg.done
echo ">>> judge leg complete"
