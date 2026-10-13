#!/usr/bin/env python3
"""Identity-probe specs: stock + the 6 RE-SERVED original adapters (DESIGN.md § Data trust item 3).

The probe proves each re-served sampler URI attaches and differs from stock before any bulk spend —
the "an original at stock level means the wrong adapter" sanity in START.md § The idea.
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--manifest", type=Path, required=True); ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args(); man = json.loads(a.manifest.read_text())
out = {"stock": {"arm": "stock", "seed": None, "training": "stock", "sampler_path": None}}
for s in man["subjects"]:
    if s["training"] == "reused":
        out[s["subject"]] = {"arm": s["arm"], "seed": s["seed"], "training": "reused",
                             "row_family": s.get("row_family"), "arm_kind": s.get("arm_kind"),
                             "sampler_path": s["sampler_path"]}
assert len(out) == 7, len(out)
assert all(out[x].get("sampler_path", "").startswith("tinker://") for x in out if x != "stock")
a.out.parent.mkdir(parents=True, exist_ok=True); a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(">>> identity probe specs: stock + 6 re-served original adapters")
