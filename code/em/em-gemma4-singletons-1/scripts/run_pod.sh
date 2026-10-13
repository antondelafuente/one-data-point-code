#!/bin/bash
# Outer wrapper: run job.sh detached with its log, then upload the log + exit code and write DONE.
EXP=em-gemma4-singletons-1; P=/workspace/$EXP; R2=remote:bucket/experiments/$EXP
[ "${R2##*/}" = "$EXP" ] || exit 1
cd $P; bash scripts/job.sh > $P/job.log 2>&1; rc=$?
echo $rc > $P/exit_code.txt
for f in job.log exit_code.txt stages.txt setup.log setup_job.log; do [ -f $P/$f ] && rclone copyto --retries 15 $P/$f $R2/pods/$RUNPOD_POD_ID/$f; done
rclone check --one-way --size-only --include '{job.log,exit_code.txt,stages.txt}' $P $R2/pods/$RUNPOD_POD_ID/ > $P/final_check.txt 2>&1 && touch $P/DONE_VERIFIED
touch $P/DONE
