#!/usr/bin/env bash
# Launch a driver detached and record its REAL pid (the wrapper bash execs into the target, so
# $$ is the target's pid). `setsid nohup cmd & echo $!` records the setsid wrapper instead, which
# exits immediately and makes every later `kill -0` liveness poll read DEAD while the job is fine.
# usage: launch_detached.sh <pidfile> <logfile> <cmd...>
set -euo pipefail
PIDFILE="$1"; LOGFILE="$2"; shift 2
mkdir -p "$(dirname "$PIDFILE")" "$(dirname "$LOGFILE")"
setsid bash -c 'echo $$ > "$1"; exec "${@:3}" >> "$2" 2>&1' _ "$PIDFILE" "$LOGFILE" "$@" &
sleep 1
echo "launched pid $(cat "$PIDFILE") -> $LOGFILE"
