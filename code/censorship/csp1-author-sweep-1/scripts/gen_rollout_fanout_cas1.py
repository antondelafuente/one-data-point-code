#!/usr/bin/env python3
"""cas1: generate the rollout + gsm8k command files for local_job_queue.sh.
22 rollout subjects (12 trained + 9 anchors + stock), profile cas1full (B0 90x20 + B1 60x5 +
B2 60x5 = 2,400 rows each), + 12 GSM8K-200 jobs (trained subjects only).
The canonical expected-subject list (#337) is emitted to work/expected_subjects.json."""
import json, os

REPO = os.path.expanduser("~/ws/run/csp1-author-sweep-1")
REG = f"{REPO}/registry"
W = f"{REG}/csp1-author-sweep-1/work"
PY = "/path/to/home/venvs/tinker-cd1/bin/python"
GEN = f"{W}/scripts/generate_rollouts_cas1.py"
GSM = f"{W}/scripts/gsm8k_cas1.py"
BASE = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

subjects = {}  # label -> {kind, model_path or base_model}
cas1 = json.load(open(f"{REG}/csp1-author-sweep-1/cas1_sampler_paths.json"))
assert len(cas1) == 12
for name, info in cas1.items():
    subjects[f"cas1_{name}"] = {"kind": "arm", "model_path": info["sampler_path"]}

sb1 = json.load(open(f"{REG}/csp1-selfbase-1/sb1_sampler_paths.json"))
for k in ("sb1_floor_seed3", "sb1_floor_seed4", "sb1_orig_seed3", "sb1_orig_seed4"):
    subjects[k] = {"kind": "reeval", "model_path": sb1[k]["sampler_path"]}
dl1 = json.load(open(f"{REG}/csp1-refusal-dose-ladder-1/dl1_sampler_paths.json"))
for k in ("D_strict_dfull_s3", "D_strict_dfull_s4"):
    subjects[f"dl1_{k}"] = {"kind": "reeval", "model_path": dl1[k]["sampler_path"]}
mdc = json.load(open(f"{REG}/csp1-maxdose-cells-1/csp1mdc1_sampler_paths.json"))
for k in ("A_selfmax_s3", "A_selfmax_s4", "A_selfmax_s5"):
    subjects[f"mdc1_{k}"] = {"kind": "reeval", "model_path": mdc[k]["sampler_path"]}
subjects["stock"] = {"kind": "stock", "base_model": BASE}

assert len(subjects) == 22, len(subjects)
json.dump({"subjects": {k: {"kind": v["kind"]} for k, v in subjects.items()},
           "per_subject_rows": {"B0": 1800, "B1": 300, "B2": 300},
           "gsm8k_subjects": sorted(k for k, v in subjects.items() if v["kind"] == "arm")},
          open(f"{W}/expected_subjects.json", "w"), indent=1)

lines = []
for label, cfg in sorted(subjects.items()):
    src = f"--model-path {cfg['model_path']}" if "model_path" in cfg else f"--base-model {cfg['base_model']}"
    lines.append(f"cd {REPO} && {PY} {GEN} --label {label} --profile cas1full {src} "
                 f"--out {W}/rollouts/{label}.jsonl --concurrency 32 "
                 f"> {W}/logs/gen_{label}.log 2>&1")
open(f"{W}/rollout_fanout_cmds.txt", "w").write("\n".join(lines) + "\n")

glines = []
for label, cfg in sorted(subjects.items()):
    if cfg["kind"] != "arm":
        continue
    glines.append(f"cd {REPO} && {PY} {GSM} --label {label} --model-path {cfg['model_path']} "
                  f"--out {W}/gsm8k/{label}_gsm8k.jsonl "
                  f"> {W}/logs/gsm8k_{label}.log 2>&1")
open(f"{W}/gsm8k_fanout_cmds.txt", "w").write("\n".join(glines) + "\n")
print(f"wrote {len(lines)} rollout cmds + {len(glines)} gsm8k cmds; expected_subjects.json emitted")
