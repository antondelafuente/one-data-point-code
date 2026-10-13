#!/usr/bin/env bash
# The bulk leg for ONE stage: ONE serving session for all of that stage's subjects (rolling groups
# of <=6 hot samplers, work-stealing driver lanes), 12 scenarios x 3 draws x 8 turns per subject;
# then the frozen mechanical degen screen, the strict-despair judge with its quote-validation loop,
# the blind Luna coherence classifier over EVERY raw conversation, and the frozen MECHANICAL
# COHERENCE RULE (this wave's primary coherence read, DESIGN.md § Model choices).
#
# Generalised from depv1-negemo-qwen-chat-row11-dose-placebo-1/scripts/run_main.sh: everything is
# per-stage (rollouts/$STAGE, judge/$STAGE, coherence/*_$STAGE.jsonl, session id $STAGE) because
# this wave has two serving sessions — Stage 1 (24 candidates at seed 4) and, after the designer's
# committed selection, Stage 2 (the selected rows at seeds 5 and 6).
#
# Idempotent and resume-aware at every step. Detached: poll logs/.$STAGE.done.
set -uo pipefail
W="${W:?set W to the experiment work directory}"
E="${E:?set E to the experiment registry directory}"
STAGE="${STAGE:-main}"
PY=${PY:-/path/to/home/venvs/tinker-cd1/bin/python}
CONC=${CONC:-96}
JUDGE_CONC=${JUDGE_CONC:-48}
LUNA_CONC=${LUNA_CONC:-50}
cd "$W"; mkdir -p logs judge/"$STAGE" coherence rollouts/"$STAGE" analysis
echo $$ > logs/stage_$STAGE.pid          # this driver's OWN pid, for liveness polling by PID (#462)
LOG=logs/$STAGE.log
BAT=inputs/selfcontempt_battery_min_v1.json
MANIFEST=subject_manifest_$STAGE.json
GROUPS_FILE=serving_groups_$STAGE.json   # NOT `GROUPS`: bash special var (user gids)

step () { echo ">>> [$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

NSUB=$("$PY" -c "import json;print(len(json.load(open('$MANIFEST'))['eval_subjects']))")
NCONV=$(( NSUB * 36 ))
step "$STAGE: $NSUB subjects, one serving session, $NCONV conversations"

if [ ! -s rollouts/"$STAGE"/generation_verification.json ]; then
  "$PY" "$E/scripts/run_generation.py" --work "$W" --groups "$GROUPS_FILE" \
      --manifest "$MANIFEST" --battery "$BAT" --leg main --concurrency "$CONC" \
      --session-id "$STAGE" --draws 0,1,2 --outdir rollouts/"$STAGE" \
      --orchestration-out rollouts/"$STAGE"/orchestration.json >> "$LOG" 2>&1
  echo ">>> generation orchestrator exit=$?" | tee -a "$LOG"
fi

step "trajectories + frozen mechanical degen screen (labels, never deletes)"
"$PY" "$E/scripts/build_trajectories.py" \
    --rollouts rollouts/"$STAGE"/g*.jsonl \
    --battery "$BAT" --leg main --subjects "$MANIFEST" \
    --degen "$E/scripts/degen.py" --draws 0,1,2 \
    --out rollouts/"$STAGE"/trajectories.jsonl \
    --report rollouts/"$STAGE"/trajectory_report.json \
    >> "$LOG" 2>&1 || { echo ">>> BLOCKED: trajectory assembly failed" | tee -a "$LOG"; exit 1; }

step "fail-closed generation verification"
"$PY" "$E/scripts/verify_generation.py" --dir rollouts/"$STAGE" --manifest "$MANIFEST" \
    --groups "$GROUPS_FILE" --session-id "$STAGE" 2>&1 | tail -40 | tee -a "$LOG"
genrc=${PIPESTATUS[0]}
if [ "$genrc" -ne 0 ]; then echo ">>> BLOCKED: generation verification FAILED" | tee -a "$LOG"; exit 2; fi

cp -f rollouts/"$STAGE"/trajectories.jsonl judge/"$STAGE"/main_input.jsonl

step "strict despair + quote validation (backgrounded), blind coherence, mechanical rule"
E="$E" W="$W" PY="$PY" CONC="$JUDGE_CONC" STAGE="$STAGE" nohup setsid bash "$E/scripts/run_judge.sh" \
    </dev/null > logs/judge_driver_$STAGE.log 2>&1 &
echo "$!" > logs/judge_driver_$STAGE.pid
# `$!` can be the setsid WRAPPER rather than the judge driver, so liveness is polled against the pid
# run_judge.sh writes for itself; the marker is the terminal condition either way (#462).
for _ in $(seq 1 60); do [ -s logs/judge_self_$STAGE.pid ] && break; sleep 1; done
judge_pid=$(cat logs/judge_self_$STAGE.pid 2>/dev/null || echo 0)

"$PY" "$E/scripts/mech_coherence_pass.py" --rollouts rollouts/"$STAGE"/g*.jsonl \
    --trajectories rollouts/"$STAGE"/trajectories.jsonl \
    --rule "$E/scripts/mech_coherence_rule.py" \
    --out coherence/mech_rule_$STAGE.jsonl \
    --report coherence/mech_rule_report_$STAGE.json >> "$LOG" 2>&1
mechrc=$?
echo ">>> mechanical coherence rule exit=$mechrc" | tee -a "$LOG"

"$PY" "$E/scripts/classify_coherence.py" --input rollouts/"$STAGE"/trajectories.jsonl \
    --output coherence/luna_$STAGE.jsonl --model gpt-5.6-luna --rubric-version v3 \
    --concurrency "$LUNA_CONC" --hard-stop-usd 25.0 >> "$LOG" 2>&1
echo ">>> luna coherence exit=$?" | tee -a "$LOG"

step "waiting on the despair judge leg"
deadline=$(( $(date +%s) + 14400 ))
while [ ! -f logs/.judged_$STAGE.done ]; do
  kill -0 "$judge_pid" 2>/dev/null || { sleep 20; [ -f logs/.judged_$STAGE.done ] && break; \
      echo ">>> judge pid $judge_pid gone without a marker" | tee -a "$LOG"; break; }
  [ "$(date +%s)" -lt "$deadline" ] || { echo ">>> judge wait deadline passed" | tee -a "$LOG"; break; }
  sleep 30
done
[ -f logs/.judged_$STAGE.done ] || { echo ">>> BLOCKED: judge leg exited without its marker" | tee -a "$LOG"; exit 3; }

step "fail-closed label verification"
"$PY" "$E/scripts/verify_labels.py" --work "$W" --stage "$STAGE" --expected "$NCONV" \
    2>&1 | tail -40 | tee -a "$LOG"
labrc=${PIPESTATUS[0]}

touch logs/.$STAGE.done
echo ">>> $STAGE leg finished gen_rc=$genrc mech_rc=$mechrc label_rc=$labrc $(date -u +%FT%TZ)" | tee -a "$LOG"
exit "$labrc"
