#!/bin/bash
# csp1-matched-prompt-teacher-swap-1: Task-2 judge pass over all 9 subjects' rollout files.
# DESIGN.md pins "aggregate API concurrency 90" total -- split evenly across 9 subjects (--conc
# 10 each) so all 9 can run concurrently within budget, pipelined per subject as its rollout file
# reaches its full row count (this driver polls for that, launching each subject's judge job the
# moment its rollout file is complete -- "start each subject's Task-2 judge fan-out as soon as
# that subject's rollout file completes"). judge_task2_mpts1.py is judge_task2_hbe1.py
# (csp1-harmful-boundary-eval-1) with ONLY its hardcoded REG path repointed at THIS experiment's
# own registry dir -- the original script resolves task2_prompt.txt via
# Path.home()/"ws/run/csp1-harmful-boundary-eval-1/registry/..." which does not exist as a live
# worktree on this box (that experiment closed), so running it unmodified crashed all 9 judge
# jobs at import time on the first launch attempt. Both task2_prompt.txt files are byte-identical
# (sha256 f2a8bd5c...c8602ce88f36 matches DESIGN.md's pinned value) -- pure path fix, same tiered
# never-inventing parser incl. the #542 bare-text fallback, same semaphore-per-attempt shape
# avoiding the #343 stuck-backoff stall. Each subject writes its OWN judged/<subject>.jsonl.
# Resumable via --key-fields dedup.
set -uo pipefail
cd /path/to/home/ws/run/csp1-matched-prompt-teacher-swap-1
export OPENAI_API_KEY
PY=/path/to/home/venvs/tinker-cd1/bin/python
JUDGE=registry/csp1-matched-prompt-teacher-swap-1/scripts/judge_task2_mpts1.py
ROLLOUTS=/path/to/home/work/csp1-matched-prompt-teacher-swap-1/rollouts
OUT=/path/to/home/work/csp1-matched-prompt-teacher-swap-1/judged
LOGS=/path/to/home/work/csp1-matched-prompt-teacher-swap-1/logs
SUBJECTS_FILE=/path/to/home/work/csp1-matched-prompt-teacher-swap-1/subjects.json
ROWS_PER_SUBJECT=3070
mkdir -p "$OUT" "$LOGS"

subjects=$("$PY" -c "import json;print(' '.join(s['subject'] for s in json.load(open('$SUBJECTS_FILE'))))")
n_subj=$(echo "$subjects" | wc -w)
if [ "$n_subj" -ne 9 ]; then
  echo "FATAL: expected 9 subjects in $SUBJECTS_FILE, got $n_subj" >&2
  exit 1
fi

echo "watching rollout files for completion (>= $ROWS_PER_SUBJECT rows each), launching judge per subject as ready"
declare -A launched
while true; do
  all_launched=1
  for subj in $subjects; do
    if [ -n "${launched[$subj]:-}" ]; then continue; fi
    all_launched=0
    f="$ROLLOUTS/${subj}.jsonl"
    if [ -f "$f" ]; then
      n=$(wc -l < "$f" 2>/dev/null || echo 0)
      if [ "$n" -ge "$ROWS_PER_SUBJECT" ]; then
        log="$LOGS/judge_${subj}.log"
        echo "launching judge for $subj ($n rows ready)"
        nohup "$PY" "$JUDGE" --in "$f" --out "$OUT/${subj}.jsonl" \
          --key-fields model,battery,id,sample --model gpt-5.4-mini-2026-03-17 \
          --reasoning-effort low --conc 10 \
          > "$log" 2>&1 &
        launched[$subj]=1
      fi
    fi
  done
  if [ "$all_launched" -eq 1 ]; then
    echo "all $n_subj judge jobs launched"
    break
  fi
  sleep 20
done
wait
echo "ALL $n_subj JUDGE JOBS FINISHED" | tee "$LOGS/judge_fanout_ALL_DONE.marker"
