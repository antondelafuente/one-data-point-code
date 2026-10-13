#!/usr/bin/env python3
"""CHAT LoRA target census (BLOCK gate).

DESIGN: "Do not assume the Base checkpoint's 248-module target count. Probe chat Qwen first,
freeze the observed module-name histogram, and require every real train to match that chat-model
census."

So this script has NO expected histogram baked in. It downloads a rank-32 Tinker LoRA adapter
archive trained on `Qwen/Qwen3.5-9B` (chat), reads the tensor keys, derives the logical target-module
list + a suffix histogram, and writes chat_target_census.json. The FIRST run freezes the census;
every later run is a re-probe that must MATCH the frozen file (that is the "every real train matches
the chat census" check, run against independently trained adapters).

The Base-model census (248) is recorded only as a contrast, never as a requirement.
"""
import collections, json, os, re, sys, tarfile, urllib.request
from pathlib import Path
import tinker

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
SAMPLER_PATH = sys.argv[2]
SUBJECT = sys.argv[3] if len(sys.argv) > 3 else "unknown"
CENSUS = Path(sys.argv[4]) if len(sys.argv) > 4 else OUT.parent / "chat_target_census.json"
BASE_QWEN_HISTOGRAM_FOR_CONTRAST = {
    "in_proj_q": 24, "in_proj_k": 24, "in_proj_v": 24, "in_proj_z": 24, "out_proj": 24,
    "q_proj": 8, "k_proj": 8, "v_proj": 8, "o_proj": 8,
    "gate_proj": 32, "up_proj": 32, "down_proj": 32,
}
SUFFIXES = sorted(BASE_QWEN_HISTOGRAM_FOR_CONTRAST)

sc = tinker.ServiceClient(timeout=3600)
print(">>> probing sampler path:", SAMPLER_PATH, flush=True)
rc = sc.create_rest_client()
url = rc.get_checkpoint_archive_url_from_tinker_path(SAMPLER_PATH).result()
u = url if isinstance(url, str) else getattr(url, "url", None) or getattr(url, "archive_url", None)
(OUT / f"archive_url_{SUBJECT}.txt").write_text(str(u) + "\n")
print(">>> archive url obtained", flush=True)
tarp = OUT / f"adapter_archive_{SUBJECT}.tar"
urllib.request.urlretrieve(u, tarp)
print(">>> downloaded", tarp.stat().st_size, "bytes", flush=True)

names = []
with tarfile.open(tarp) as tf:
    members = tf.getnames()
    (OUT / f"archive_members_{SUBJECT}.txt").write_text("\n".join(members) + "\n")
    for m in members:
        if m.endswith(".safetensors"):
            data = tf.extractfile(m).read()
            tmp = OUT / f"{SUBJECT}_{os.path.basename(m)}"
            tmp.write_bytes(data)
            from safetensors.torch import load_file
            names.extend(list(load_file(str(tmp)).keys()))
            tmp.unlink()
        elif m.endswith(".json"):
            (OUT / f"{SUBJECT}_{os.path.basename(m)}").write_bytes(tf.extractfile(m).read())
tarp.unlink()

(OUT / f"tensor_keys_{SUBJECT}.txt").write_text("\n".join(sorted(names)) + "\n")
LORA_RE = re.compile(r"^(.*?)\.lora_[AB](?:\.weight)?$")
logical = sorted({LORA_RE.match(k).group(1) for k in names if LORA_RE.match(k)})
hist, unassigned = collections.Counter(), []
for n in logical:
    m = [s for s in SUFFIXES if n.endswith("." + s) or n == s]
    if len(m) == 1:
        hist[m[0]] += 1
    else:
        unassigned.append(n)

observed = {"n_tensor_keys": len(names), "n_logical_modules": len(logical),
            "histogram": dict(sorted(hist.items())), "unassigned": sorted(unassigned),
            "logical_modules": logical}
probe = {"subject": SUBJECT, "sampler_path": SAMPLER_PATH, "base_model": "Qwen/Qwen3.5-9B",
         "lora_rank": 32, "train_mlp": True, "train_attn": True, "train_unembed": False,
         **observed,
         "base_qwen_histogram_for_contrast": BASE_QWEN_HISTOGRAM_FOR_CONTRAST,
         "matches_base_qwen_248": (len(logical) == 248
                                   and dict(hist) == BASE_QWEN_HISTOGRAM_FOR_CONTRAST)}
(OUT / f"lora_target_probe_{SUBJECT}.json").write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n")

if CENSUS.exists():
    frozen = json.loads(CENSUS.read_text())
    same = (frozen["n_logical_modules"] == observed["n_logical_modules"]
            and frozen["histogram"] == observed["histogram"]
            and frozen["logical_modules"] == observed["logical_modules"])
    probe["census_match"] = same
    probe["census_frozen_by"] = frozen["frozen_by_subject"]
    (OUT / f"lora_target_probe_{SUBJECT}.json").write_text(
        json.dumps(probe, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in probe.items() if k != "logical_modules"}, indent=2)[:2500],
          flush=True)
    print(f">>> CENSUS RE-PROBE {'MATCH' if same else 'MISMATCH'} vs frozen "
          f"({frozen['frozen_by_subject']})", flush=True)
    sys.exit(0 if same else 1)

census = {"base_model": "Qwen/Qwen3.5-9B", "lora_rank": 32, "train_mlp": True, "train_attn": True,
          "train_unembed": False, "frozen_by_subject": SUBJECT,
          "frozen_from_sampler_path": SAMPLER_PATH, **observed,
          "note": "chat-Qwen LoRA target census, OBSERVED (not inherited from the Base model's 248)"}
CENSUS.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")
print(json.dumps({k: v for k, v in probe.items() if k != "logical_modules"}, indent=2)[:2500],
      flush=True)
print(f">>> CENSUS FROZEN: {observed['n_logical_modules']} logical modules -> {CENSUS}", flush=True)
