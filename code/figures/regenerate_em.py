#!/usr/bin/env python3
"""Rebuild the numeric EM paper figures from bundled text-free results, without network or model calls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    bundle = here.parents[1] / "data" / "figure_inputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=here,
                        help="Output root; PDF, PNG and numeric JSON files are written under em/.")
    args = parser.parse_args()
    manifest = json.loads((bundle / "manifest.json").read_text())
    for entry in manifest["files"]:
        path = bundle / "registry" / entry["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise SystemExit(f"Bundled figure input checksum mismatch: {entry['path']}")
    env = dict(os.environ, MPLBACKEND="Agg", ODP_REGISTRY=str(bundle / "registry"),
               ODP_FIGURE_OUT=str(args.output.resolve()))
    for name in ("em_funnel", "em_rates", "em_coherence"):
        subprocess.run([sys.executable, str(here / "scripts" / f"{name}.py")], env=env, check=True)
    print(f"Wrote three figures to {args.output.resolve() / 'em'}")


if __name__ == "__main__":
    main()
