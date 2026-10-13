#!/usr/bin/env bash
# 24-conversation-class PREFLIGHT generation (DESIGN gate 6 + AMENDMENT A1 "include a few praise
# conversations"). Generates DRAW 0 only, for two frozen serving groups that between them cover
# stock and complete voice triplets on two row families, on BOTH batteries.
#
# The rows land in the REAL rollout files (rollouts/<leg>/gNN.jsonl) so the bulk run resumes them
# instead of paying twice; the PASS/FAIL summary is written to a preflight-only path so the bulk
# orchestrator never mistakes a draw-0 slice for a completed group.
set -uo pipefail
W="${W:-$HOME/work/depv1-negemo-qwen-chat-selfref-rewrite-1}"
PY=${PY:-$HOME/venvs/tinker-cd1/bin/python}
cd "$W"; mkdir -p logs rollouts/main rollouts/praise
for gid in g02 g05; do
  "$PY" scripts/gen_tinker.py generate --specs "rollouts/main/$gid.specs.json" \
      --battery inputs/selfcontempt_battery_min_v1.json --leg main --group-id "$gid" \
      --session-id main --driver preflight --draws 0 --concurrency 48 \
      --output "rollouts/main/$gid.jsonl" \
      --summary "rollouts/main/PREFLIGHT_$gid.summary.json" >> logs/preflight_gen.log 2>&1
  echo ">>> attack $gid exit=$?"
  "$PY" scripts/gen_tinker.py generate --specs "rollouts/praise/$gid.specs.json" \
      --battery inputs/emotion_batteries_v2.json --leg praise --group-id "$gid" \
      --session-id main --driver preflight --draws 0 --concurrency 48 \
      --output "rollouts/praise/$gid.jsonl" \
      --summary "rollouts/praise/PREFLIGHT_$gid.summary.json" >> logs/preflight_gen.log 2>&1
  echo ">>> praise $gid exit=$?"
done
touch logs/.preflight_gen.done
echo ">>> preflight generation done $(date -u +%FT%TZ)"
