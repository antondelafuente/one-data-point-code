#!/usr/bin/env python3
"""csp1-disagreement-threshold-ladder-1: check the GSM8K-200 capability gate (>=85% exact-match)
for all 8 new students. Run-validity gate, not a scientific result threshold (DESIGN)."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir containing gsm8k_<label>.jsonl per new student")
    ap.add_argument("--labels", required=True, help="json list of the 8 new-student labels")
    a = ap.parse_args()

    labels = json.loads(Path(a.labels).read_text())
    results = {}
    all_pass = True
    for label in labels:
        p = Path(a.dir) / f"gsm8k_{label}.jsonl"
        if not p.exists():
            print(f"[{label}] MISSING file {p}")
            all_pass = False
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        n = len(rows)
        correct = sum(1 for r in rows if r.get("correct"))
        pct = 100.0 * correct / n if n else 0.0
        ok = n == 200 and pct >= 85.0
        results[label] = {"n": n, "correct": correct, "pct": round(pct, 2), "pass": ok}
        print(f"[{label}] n={n} correct={correct} pct={pct:.2f}% {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False

    print(f"\nGATE {'PASS' if all_pass else 'FAIL'} (all 8 new students >=85% on GSM8K-200)")
    with open(Path(a.dir) / "gsm8k_gate_report.json", "w") as f:
        json.dump({"results": results, "gate_pass": all_pass}, f, indent=2)
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
