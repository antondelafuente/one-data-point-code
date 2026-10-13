#!/usr/bin/env bash
# Pod-side driver for depv1-negemo-union-dose-response-1. Self-contained + idempotent:
# provision -> serve -> canary -> smoke -> main generation -> upload -> done marker.
#
# Env: POD_INDEX (required), CONCURRENCY (default 24).
# Every stage skips itself if its output already exists, so a relaunch resumes.
set -uo pipefail

POD_INDEX="${POD_INDEX:?POD_INDEX required}"
CONC="${CONCURRENCY:-24}"
RUN=/workspace/pods/depv1-negemo-union-dose-response-1
R2=remote:bucket/experiments/depv1-negemo-union-dose-response-1
BASE_REV=68c46c4b3498877f3ef123c856ecfde50c39f404
BASE_DIR=/workspace/models/Qwen3.5-9B-Base
TOK_DIR=/workspace/models/Qwen3.5-9B-tokenizer
VENV=/workspace/venv-vllm-019
LOG="$RUN/logs/pod_job.log"

# The Qwen3.5 GDN linear-attention kernel JIT-compiles at first token and needs
# `ninja`, which ships INSIDE the vllm venv. Without the venv on PATH the
# EngineCore dies with FileNotFoundError: 'ninja' and every request 500s
# (the parent run hit the identical failure — its R2 keeps a
# `vllm_canary_failure_missing_path_ninja.log`). Export it before serving.
export PATH="$VENV/bin:$PATH"

mkdir -p "$RUN/inputs" "$RUN/logs" "$RUN/generation" /workspace/models
exec > >(tee -a "$LOG") 2>&1
echo "=== pod_job pod=$POD_INDEX start $(date -u +%FT%TZ) ==="

# --- [1/7] rclone + venv --------------------------------------------------
if ! command -v rclone >/dev/null; then
  curl -fsSL https://rclone.org/install.sh | bash >/dev/null 2>&1 || true
fi
# Configure the remote used by R2 without putting credentials in process arguments
# or persisting them to a config file. Existing rclone configurations remain usable.
RCLONE_REMOTE_NAME=${R2%%:*}
if ! rclone listremotes 2>/dev/null | grep -Fxq "${RCLONE_REMOTE_NAME}:"; then
  source /etc/rp_environment 2>/dev/null || true
  : "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID is required}"
  : "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY is required}"
  : "${R2_ENDPOINT:?R2_ENDPOINT is required}"
  if [[ ! "$RCLONE_REMOTE_NAME" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "BLOCKED: rclone remote name must contain only letters, digits or underscores" >&2
    exit 2
  fi
  RCLONE_ENV_PREFIX="RCLONE_CONFIG_${RCLONE_REMOTE_NAME^^}"
  export "${RCLONE_ENV_PREFIX}_TYPE=s3" "${RCLONE_ENV_PREFIX}_PROVIDER=Other"
  export "${RCLONE_ENV_PREFIX}_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID"
  export "${RCLONE_ENV_PREFIX}_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY"
  export "${RCLONE_ENV_PREFIX}_ENDPOINT=$R2_ENDPOINT"
  export "${RCLONE_ENV_PREFIX}_ACL=private" "${RCLONE_ENV_PREFIX}_NO_CHECK_BUCKET=true"
fi
if [[ ! -x "$VENV/bin/vllm" ]]; then
  echo "--- pulling venv-vllm-019 ---"
  cd /workspace && rclone copy remote:bucket/mo/venv-vllm-019.tar /workspace/ --transfers 4 \
    && tar xf /workspace/venv-vllm-019.tar -C /workspace && rm -f /workspace/venv-vllm-019.tar
fi
"$VENV/bin/vllm" --version || { echo "BLOCKED: vllm venv unusable"; exit 2; }

# The archived venv ships `bin/ninja` as a SYMLINK into /workspace/venv, a
# sibling venv this run never pulls — so it dangles, and flashinfer's JIT dies
# with FileNotFoundError: 'ninja' at first token, killing EngineCore. Replace
# the dangling link with a real ninja inside this venv.
if ! "$VENV/bin/ninja" --version >/dev/null 2>&1; then
  echo "--- repairing ninja in $VENV ---"
  rm -f "$VENV/bin/ninja"
  # --force-reinstall: the `ninja` wheel is already in site-packages, so a plain
  # install is a no-op and never recreates the console script we just deleted.
  "$VENV/bin/pip" install -q --no-input --force-reinstall ninja || true
  if ! "$VENV/bin/ninja" --version >/dev/null 2>&1; then
    # Fall back to the binary the wheel ships inside site-packages.
    src=$(find "$VENV/lib" -path '*/ninja/data/bin/ninja' -type f 2>/dev/null | head -1)
    [[ -n "$src" ]] && ln -sf "$src" "$VENV/bin/ninja"
  fi
  "$VENV/bin/ninja" --version || { echo "BLOCKED: ninja unavailable"; exit 2; }
fi

# --- [2/7] base model + tokenizer ----------------------------------------
if [[ ! -f "$BASE_DIR/.base_complete" ]]; then
  echo "--- downloading base @ $BASE_REV ---"
  HF_HUB_DISABLE_XET=1 "$VENV/bin/hf" download Qwen/Qwen3.5-9B-Base \
    --revision "$BASE_REV" --local-dir "$BASE_DIR" || { echo "BLOCKED: base download"; exit 3; }
  touch "$BASE_DIR/.base_complete"
fi
rm -rf "$TOK_DIR"; mkdir -p "$TOK_DIR"
find "$BASE_DIR" -maxdepth 1 -type f \
  \( -name '*token*' -o -name 'merges.txt' -o -name 'vocab.json' \
     -o -name 'chat_template*' -o -name 'special_tokens_map.json' \) \
  -exec cp -a {} "$TOK_DIR/" \;

# --- [3/7] this run's inputs + the adapters this pod serves ---------------
# Pods are acquired and provisioned (venv + 18 GB base model, ~25 min) IN PARALLEL
# with the box-side Tinker export/convert leg, so the adapters may not be in R2 yet
# when we get here. `pod_inputs/.inputs_complete` is written by the box only after
# every converted adapter + adapter_sha256.json has landed, so it is the one safe
# gate to wait on. Bounded wait; a miss is a loud BLOCK, never a silent skip.
# NOTE: `rclone lsf <remote>/<missing-object>` exits 0 with EMPTY output on S3 —
# it reads the path as an empty directory listing — so the marker check must test
# the OUTPUT, never the exit code. (Caught live: two pods sailed straight past an
# exit-code-only version of this gate and died on the adapter hash check instead.)
marker_present() { [[ -n "$(rclone lsf "$R2/pod_inputs/" 2>/dev/null | grep -Fx '.inputs_complete')" ]]; }
for i in $(seq 1 240); do   # 240 x 30s = 2h ceiling
  marker_present && break
  [[ $((i % 10)) -eq 1 ]] && echo "  waiting for box-side adapters in R2 ($i) $(date -u +%FT%TZ)"
  sleep 30
done
marker_present \
  || { echo "BLOCKED: pod_inputs/.inputs_complete never appeared in R2"; exit 9; }
rclone copy "$R2/pod_inputs/" "$RUN/inputs/" --transfers 8 --checkers 8
cp "$RUN/inputs/tokenizer/tokenizer_config.json" "$TOK_DIR/tokenizer_config.json"
cp "$RUN/inputs/tokenizer/chat_template.jinja" "$TOK_DIR/chat_template.jinja"

# grep -v '^$': a base-only pod prints nothing, and a bare `mapfile` on that
# still yields ONE empty-string element — which made the guard below emit a
# bare `--lora-modules` and vLLM refused to start ("Both 'lora_name' and
# 'lora_path' must be provided").
mapfile -t ADAPTERS < <("$VENV/bin/python" -c "
import json,sys
plan=json.load(open('$RUN/inputs/cells.json'))
print('\n'.join(sorted({c['adapter'] for c in plan['cells'] if c['pod']==$POD_INDEX and c['adapter']!='base'})))
" | grep -v '^$')
echo "--- pod $POD_INDEX adapters: ${ADAPTERS[*]:-<base only>} ---"
for a in "${ADAPTERS[@]}"; do
  [[ -f "$RUN/inputs/adapters/$a/adapter_model.safetensors" ]] || \
    rclone copy "$R2/adapters/converted/$a/" "$RUN/inputs/adapters/$a/" --transfers 4
done
# Verify every adapter against its committed converted hash before serving it.
"$VENV/bin/python" - <<PYEOF || { echo "BLOCKED: adapter hash mismatch"; exit 4; }
import hashlib, json, sys, pathlib
want = json.load(open("$RUN/inputs/adapter_sha256.json"))
bad = []
for a in """${ADAPTERS[*]:-}""".split():
    p = pathlib.Path("$RUN/inputs/adapters")/a/"adapter_model.safetensors"
    got = hashlib.sha256(p.read_bytes()).hexdigest()
    if got != want[a]:
        bad.append((a, got, want[a]))
print("adapter hash check:", "OK" if not bad else bad)
sys.exit(1 if bad else 0)
PYEOF

# --- [4/7] serve ----------------------------------------------------------
serve_up() { curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; }
if ! serve_up; then
  ARGV=( "$VENV/bin/vllm" serve "$BASE_DIR"
    --served-model-name qwen3.5-9b-base
    --tokenizer "$TOK_DIR" --host 127.0.0.1 --port 8000
    --dtype bfloat16 --language-model-only --max-model-len 65536
    --enable-lora --max-loras 4 --max-lora-rank 128 )
  if (( ${#ADAPTERS[@]} > 0 )); then
    ARGV+=( --lora-modules )
    for a in "${ADAPTERS[@]}"; do ARGV+=( "$a=$RUN/inputs/adapters/$a" ); done
  fi
  printf '%q ' "${ARGV[@]}" > "$RUN/generation/server_argv_pod${POD_INDEX}.txt"; printf '\n' >> "$RUN/generation/server_argv_pod${POD_INDEX}.txt"
  nohup setsid "${ARGV[@]}" > "$RUN/logs/vllm_pod${POD_INDEX}.log" 2>&1 < /dev/null &
  echo $! > "$RUN/generation/vllm.pid"
  for i in $(seq 1 180); do serve_up && break; sleep 10; done
fi
serve_up || { echo "BLOCKED: vllm never came up"; tail -40 "$RUN/logs/vllm_pod${POD_INDEX}.log"; exit 5; }
echo "--- server up $(date -u +%FT%TZ) ---"

# --- [5/7] canary + smoke -------------------------------------------------
export NEGEMO_ROOT="$RUN" NEGEMO_TOKENIZER_DIR="$TOK_DIR"
if [[ ! -f "$RUN/generation/adapter_canary.json" ]]; then
  "$VENV/bin/python" "$RUN/inputs/generate_cells.py" canary --pod "$POD_INDEX" \
    || { echo "BLOCKED: canary failed"; exit 6; }
fi
if [[ ! -f "$RUN/generation/.smoke_pod${POD_INDEX}_done" ]]; then
  "$VENV/bin/python" "$RUN/inputs/generate_cells.py" smoke --pod "$POD_INDEX" --concurrency 8 \
    || { echo "BLOCKED: smoke failed"; exit 7; }
  rclone copy "$RUN/generation/" "$R2/generation/" --include "smoke_pod${POD_INDEX}*" --include "adapter_canary*"
fi
echo "--- smoke ok $(date -u +%FT%TZ) ---"

# --- [6/7] main generation ------------------------------------------------
if [[ ! -f "$RUN/generation/.generation_done_pod${POD_INDEX}" ]]; then
  "$VENV/bin/python" "$RUN/inputs/generate_cells.py" main --pod "$POD_INDEX" --concurrency "$CONC"
  rc=$?
  rclone copy "$RUN/generation/" "$R2/generation/" --include "*pod${POD_INDEX}*" --transfers 4
  rclone copy "$RUN/logs/" "$R2/logs/pod${POD_INDEX}/" --transfers 4
  [[ $rc -eq 0 ]] || { echo "BLOCKED: main generation rc=$rc"; exit 8; }
fi

# --- [7/7] upload + done --------------------------------------------------
rclone copy "$RUN/generation/" "$R2/generation/" --include "*pod${POD_INDEX}*" --include "adapter_canary*" --transfers 4
rclone copy "$RUN/logs/" "$R2/logs/pod${POD_INDEX}/" --transfers 4
rclone check "$RUN/generation/" "$R2/generation/" --include "*pod${POD_INDEX}*" --one-way \
  && echo "R2 CHECK OK pod=$POD_INDEX"
touch "$RUN/generation/.pod_job_done"
echo "=== POD $POD_INDEX DONE $(date -u +%FT%TZ) ==="
