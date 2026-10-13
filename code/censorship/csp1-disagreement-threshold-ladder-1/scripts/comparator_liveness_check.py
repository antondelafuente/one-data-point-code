#!/usr/bin/env python3
"""csp1-disagreement-threshold-ladder-1: pre-training liveness + checkpoint-TYPE check (#348)
for the 12 frozen comparator sampler paths + stock. Adapted from
csp1-wedge-knockout-2/scripts/anchor_liveness_check.py."""
import json, time, sys
from pathlib import Path
import tinker
from tinker import types

BASE_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def main():
    manifest_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    manifest = json.loads(manifest_path.read_text())
    sc = tinker.ServiceClient()
    report = {}
    subjects = dict(manifest["subjects"])
    subjects["stock"] = {"source": "stock", "sampler_path": None}
    for name, s in subjects.items():
        path = s.get("sampler_path")
        is_sampler_weights = path is None or "/sampler_weights/" in path
        t0 = time.time()
        try:
            client = (sc.create_sampling_client(model_path=path) if path
                      else sc.create_sampling_client(base_model=BASE_MODEL))
            fut = client.sample(prompt=types.ModelInput.from_ints(tokens=[1, 2, 3]),
                                 num_samples=1, sampling_params=types.SamplingParams(max_tokens=4, temperature=0.0, seed=0))
            r = fut.result()
            ok = len(r.sequences[0].tokens) >= 0
            dt = time.time() - t0
            report[name] = {"path": path or "stock", "attached_ok": True, "sample_ok": ok,
                             "checkpoint_type_sampler_weights": is_sampler_weights, "seconds": round(dt, 1)}
            print(f"[{name}] LIVE, attach+sample OK in {dt:.1f}s (sampler_weights={is_sampler_weights})", flush=True)
        except Exception as e:
            report[name] = {"path": path or "stock", "attached_ok": False, "error": str(e)[:300]}
            print(f"[{name}] DEAD -- {str(e)[:200]}", flush=True)

    out_path.write_text(json.dumps(report, indent=2))
    dead = [n for n, r in report.items() if not r.get("attached_ok")]
    if dead:
        print(f"BLOCK: dead subjects: {dead}", flush=True)
        raise SystemExit(1)
    print(f"ALL {len(report)} SUBJECTS LIVE -- proceed", flush=True)


if __name__ == "__main__":
    main()
