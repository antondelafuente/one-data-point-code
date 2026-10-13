#!/bin/bash
# collect.sh POD_ID — after a pod DONE: verify its R2 legs against the pod (pod-side check), pull light artifacts locally, tear down. Adapted from the parent.
set -uo pipefail
POD=$1; W=/path/to/home/work/em-gemma4-lr-probe-1; EXP=em-gemma4-lr-probe-1; R2=remote:bucket/experiments/$EXP
[ "${R2##*/}" = "$EXP" ] || exit 1
G=/path/to/home/automated-researcher/plugins/gpu-job/skills/gpu-job/scripts
read SSH TRAJ NONCE < <(python3 -c "import json;r=[json.loads(l) for l in open('$W/pods.jsonl') if json.loads(l)['pod_id']=='$POD'][0];print(r['ssh'],r['trajectory'],r['nonce'])")
IP=${SSH%:*}; PORT=${SSH#*:}; LRT=${TRAJ%_*}; TJ=$LRT/row194/seed0
O="-n -o StrictHostKeyChecking=no -o ConnectTimeout=20 -i /path/to/home/.ssh/id_ed25519"
ssh $O -p $PORT root@$IP "test -f /workspace/$EXP/DONE && cat /workspace/$EXP/exit_code.txt" > $W/collect_$TRAJ.exit || { echo "NOT DONE $TRAJ"; exit 2; }
[ "$(cat $W/collect_$TRAJ.exit)" = 0 ] || { echo "POD EXIT NONZERO $TRAJ: $(cat $W/collect_$TRAJ.exit)"; exit 3; }
# pod-side re-verification of every leg (independent of the copy step): trajectory, session final, pod files
ssh $O -p $PORT root@$IP "cd /workspace/$EXP; source /workspace/.env 2>/dev/null
 rclone copy --retries 15 out/ $R2/pods/$POD/out/ --exclude 'trajectories/**' --exclude 'sessions/**' >/dev/null 2>&1
 for x in 'out/trajectories/$TJ $R2/trajectories/$TJ' 'out/sessions/$TRAJ $R2/sessions/$TRAJ/final' 'out $R2/pods/$POD/out --exclude trajectories/** --exclude sessions/**'; do set -- \$x; rclone check --one-way --size-only \"\$@\" 2>&1 | grep -E 'differences found|matching files' | tr '\n' ' '; echo \" <- \$1\"; done" | tee $W/collect_$TRAJ.check
grep -q ' [1-9][0-9]* differences found' $W/collect_$TRAJ.check && { echo "UPLOAD MISMATCH $TRAJ"; exit 4; }
[ $(grep -c '0 differences found' $W/collect_$TRAJ.check) -eq 3 ] || { echo "CHECK INCOMPLETE $TRAJ"; exit 5; }
# local light pulls (no weights)
rclone copyto $R2/sessions/$TRAJ/final/rollouts_$TRAJ.jsonl $W/artifacts/rollouts/rollouts_$TRAJ.jsonl
rclone copy $R2/sessions/$TRAJ/final $W/artifacts/sessions/$TRAJ --exclude 'rollouts_*.jsonl'
rclone copy $R2/trajectories/$TJ $W/artifacts/trajectories/$TJ --exclude '*.safetensors' --exclude '*.pt'
N=$(wc -l < $W/artifacts/rollouts/rollouts_$TRAJ.jsonl); [ "$N" = 1800 ] || { echo "ROWCOUNT $TRAJ $N"; exit 6; }
bash $G/teardown.sh $POD $NONCE > $W/teardown_$TRAJ.log 2>&1; rc=$?
echo "{\"pod_id\":\"$POD\",\"deleted\":$(date +%s),\"teardown_rc\":$rc}" >> $W/pods.jsonl
echo "COLLECTED $TRAJ rows=$N teardown_rc=$rc"; tail -3 $W/teardown_$TRAJ.log
