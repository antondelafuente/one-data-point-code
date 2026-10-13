#!/bin/bash
# csp1-cross-teacher-carriers-1: Task-2 judge pass over all 20 subjects' rollout files, throttled
# to 4 concurrent processes at a time (via local_job_queue.sh) at concurrency 50 each per
# DESIGN.md's parallel schedule. judge_task2_hbe1.py pulled unmodified from
# csp1-harmful-boundary-eval-1 (same tiered never-inventing parser incl. the #542 bare-text
# fallback, same semaphore-per-attempt shape). Each subject writes its OWN judged/<subject>.jsonl
# (never a shared file across concurrent processes -- avoids any interleaved-append race);
# aggregation globs judged/*.jsonl. Resumable: judge_task2_hbe1.py's own --key-fields dedup skips
# already-successful rows.
#
# NOTE: judge_task2_hbe1.py hardcodes REG = ~/ws/run/csp1-harmful-boundary-eval-1/registry/
# csp1-harmful-boundary-eval-1 (its own home worktree layout) to read task2_prompt.txt + assert
# its sha. That path doesn't exist standalone in this worktree (HBE1's registry only exists as a
# subdirectory of THIS worktree) -- a symlink bridges it (created once, idempotent):
#   ~/ws/run/csp1-harmful-boundary-eval-1/registry/csp1-harmful-boundary-eval-1
#     -> ~/ws/run/csp1-cross-teacher-carriers-1/registry/csp1-harmful-boundary-eval-1
set -uo pipefail
cd /path/to/home/ws/run/csp1-cross-teacher-carriers-1
export OPENAI_API_KEY
PY=/path/to/home/venvs/tinker-cd1/bin/python
JUDGE=registry/csp1-harmful-boundary-eval-1/scripts/judge_task2_hbe1.py
ROLLOUTS=/path/to/home/work/csp1-cross-teacher-carriers-1/rollouts
OUT=/path/to/home/work/csp1-cross-teacher-carriers-1/judged
LOGS=/path/to/home/work/csp1-cross-teacher-carriers-1/logs
REG=registry/csp1-cross-teacher-carriers-1
QUEUE=/path/to/home/automated-researcher/plugins/experiment-lifecycle/skills/run-experiment/scripts/local_job_queue.sh
mkdir -p "$OUT" "$LOGS"

mkdir -p ~/ws/run/csp1-harmful-boundary-eval-1/registry
if [ ! -e ~/ws/run/csp1-harmful-boundary-eval-1/registry/csp1-harmful-boundary-eval-1 ]; then
  ln -s /path/to/home/ws/run/csp1-cross-teacher-carriers-1/registry/csp1-harmful-boundary-eval-1 \
    ~/ws/run/csp1-harmful-boundary-eval-1/registry/csp1-harmful-boundary-eval-1
fi

subjects=$("$PY" -c "import json;print(' '.join(s['subject'] for s in json.load(open('$REG/design_inputs/subjects.json'))))")
n_subj=$(echo "$subjects" | wc -w)
if [ "$n_subj" -ne 20 ]; then
  echo "FATAL: expected 20 subjects in subjects.json, got $n_subj" >&2
  exit 1
fi

cmdfile=$(mktemp)
for subj in $subjects; do
  log="$LOGS/judge_${subj}.log"
  echo "'$PY' '$JUDGE' --in '$ROLLOUTS/${subj}.jsonl' --out '$OUT/${subj}.jsonl' --key-fields model,battery,id,sample --model gpt-5.4-mini-2026-03-17 --reasoning-effort low --conc 50 > '$log' 2>&1" >> "$cmdfile"
done

echo "throttled-launching 20 judge jobs, cap=4 concurrent, via local_job_queue.sh"
bash "$QUEUE" 4 "$cmdfile" 10
echo "all 20 judge jobs launched (throttled dispatch complete; jobs continue running in background)"
