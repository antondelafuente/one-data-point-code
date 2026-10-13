#!/bin/bash
# em-gemma4-lr-probe-1 pod environment setup (idempotent). Writes /workspace/em-gemma4-lr-probe-1/env_ready on success.
set -euo pipefail
export HF_HUB_DISABLE_XET=1 HF_HOME=/workspace/hf
P=/workspace/em-gemma4-lr-probe-1; mkdir -p $P/out
source /workspace/.env 2>/dev/null || true
mkdir -p /root/.config/rclone
cat > /root/.config/rclone/rclone.conf <<RC
[r2]
type = s3
provider = Other
access_key_id = $R2_ACCESS_KEY_ID
secret_access_key = $R2_SECRET_ACCESS_KEY
endpoint = $R2_ENDPOINT
acl = private
no_check_bucket = true
RC
chmod 600 /root/.config/rclone/rclone.conf
[ -x $P/venv/bin/python ] || python3 -m venv --system-site-packages $P/venv
$P/venv/bin/pip install -q --disable-pip-version-check 'transformers==5.18.0' 'peft==0.21.1' 'accelerate==1.15.0' 'tokenizers==0.23.2' 'huggingface_hub==1.33.0' 'safetensors==0.8.0' 'sentencepiece==0.2.2' jinja2 'tiktoken==0.14.0' 'pillow==11.0.0'
$P/venv/bin/pip freeze > $P/out/requirements-frozen.txt
$P/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p=snapshot_download('google/gemma-4-12B-it',revision='707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7')
print('snapshot',p)
PY
touch $P/env_ready
echo SETUP_DONE
