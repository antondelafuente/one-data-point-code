#!/usr/bin/env python3
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--gate", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
gate = json.loads(a.gate.read_text())
ids = sorted({x["id"] for x in gate.get("quote_ungrounded_all", gate["quote_ungrounded"])})
a.out.write_text(json.dumps(ids, indent=2) + "\n")
print(f">>> {len(ids)} quote-validation retry ids")
