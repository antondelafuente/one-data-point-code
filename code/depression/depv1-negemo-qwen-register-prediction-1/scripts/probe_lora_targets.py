#!/usr/bin/env python3
"""BLOCK gate probe: download a rank-32 Tinker LoRA adapter archive for
Qwen/Qwen3.5-9B-Base and report the logical target-module count + histogram, so it
can be compared against the discovery 248-module split-QKV/GDN expectation."""
import json, os, re, sys, tarfile, urllib.request, collections
from pathlib import Path
import tinker

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

EXPECTED_TARGET_MODULES = 248
EXPECTED_TARGET_HISTOGRAM = {
    "in_proj_q": 24, "in_proj_k": 24, "in_proj_v": 24, "in_proj_z": 24, "out_proj": 24,
    "q_proj": 8, "k_proj": 8, "v_proj": 8, "o_proj": 8,
    "gate_proj": 32, "up_proj": 32, "down_proj": 32,
}

sc = tinker.ServiceClient(timeout=3600)
tc = sc.create_lora_training_client(base_model="Qwen/Qwen3.5-9B-Base", rank=32, train_unembed=False)
print(">>> created training client", flush=True)
r = tc.save_weights_for_sampler(name="targetprobe_sw").result()
print(">>> save_weights_for_sampler:", r.path, flush=True)
rc = sc.create_rest_client()
url = rc.get_checkpoint_archive_url_from_tinker_path(r.path).result()
print(">>> archive url obtained", flush=True)
u = url if isinstance(url, str) else getattr(url, "url", None) or getattr(url, "archive_url", None)
print(">>> url type", type(url), flush=True)
(OUT / "archive_url.txt").write_text(str(u) + "\n")
tarp = OUT / "adapter_archive.tar"
urllib.request.urlretrieve(u, tarp)
print(">>> downloaded", tarp.stat().st_size, "bytes", flush=True)

names = []
with tarfile.open(tarp) as tf:
    members = tf.getnames()
    (OUT / "archive_members.txt").write_text("\n".join(members) + "\n")
    print(">>> members:", len(members), flush=True)
    for m in members:
        if m.endswith(".safetensors"):
            f = tf.extractfile(m)
            data = f.read()
            (OUT / os.path.basename(m)).write_bytes(data)
            from safetensors.torch import load_file
            import tempfile
            tmp = OUT / os.path.basename(m)
            keys = list(load_file(str(tmp)).keys())
            names.extend(keys)
        if m.endswith(".json"):
            f = tf.extractfile(m)
            (OUT / os.path.basename(m)).write_bytes(f.read())

(OUT / "tensor_keys.txt").write_text("\n".join(sorted(names)) + "\n")
LORA_RE = re.compile(r"^(.*?)\.lora_[AB](?:\.weight)?$")
logical = sorted({LORA_RE.match(k).group(1) for k in names if LORA_RE.match(k)})
hist = collections.Counter()
unassigned = []
for n in logical:
    m = [s for s in EXPECTED_TARGET_HISTOGRAM if n.endswith("." + s) or n.endswith(s)]
    if len(m) == 1:
        hist[m[0]] += 1
    else:
        unassigned.append(n)
res = {
    "n_tensor_keys": len(names),
    "n_logical_modules": len(logical),
    "histogram": dict(hist),
    "unassigned": unassigned[:40],
    "expected_modules": EXPECTED_TARGET_MODULES,
    "expected_histogram": EXPECTED_TARGET_HISTOGRAM,
    "match_count": len(logical) == EXPECTED_TARGET_MODULES,
    "match_histogram": dict(hist) == EXPECTED_TARGET_HISTOGRAM,
    "sample_keys": sorted(names)[:30],
    "sample_logical": logical[:30],
}
(OUT / "lora_target_probe.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
print(json.dumps(res, indent=2)[:4000], flush=True)
print(">>> PROBE DONE", flush=True)
