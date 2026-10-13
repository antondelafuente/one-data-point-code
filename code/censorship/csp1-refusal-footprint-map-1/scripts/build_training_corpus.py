#!/usr/bin/env python3
"""Assemble every training-prompt corpus behind the 45 subjects, for the leakage screen."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

csv.field_size_limit(sys.maxsize)
WORK = Path(os.environ.get("RFPM1_WORK",
                           str(Path.home() / "work" / "csp1-refusal-footprint-map-1")))
REG = Path(__file__).resolve().parents[1].parent
NSSS1 = Path.home() / "work" / "csp1-nonstate-suppression-sufficiency-1"
CORP = WORK / "corpora"


def normalize(t: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t.strip())).casefold()


def main() -> None:
    rows: list[dict] = []

    def add(source: str, prompts):
        n = 0
        for p in prompts:
            if p and p.strip():
                rows.append({"source": source, "prompt": p})
                n += 1
        print(f"  {source}: {n}", flush=True)

    add("sb1_filler", (json.loads(l)["prompt"] for l in (WORK / "inputs" / "train_sb_floor.jsonl").open() if l.strip()))
    add("nscs1_nonstate500", (r["prompt"] for r in json.loads((NSSS1 / "pool" / "master500.json").read_text())))
    add("cbd1_clean56", (r["prompt"] for r in json.loads((REG / "csp1-clean-benign-dose-1" / "figures" / "carriers_56.json").read_text())))
    for rec, label in (("csp1-taboo-carrier-sufficiency-1", "tcs1_taboo500"),
                       ("csp1-generic-taboo-carrier-1", "gtc1_v2a_gasp"),
                       ("csp1-nonsexual-taboo-carrier-1", "ntc1_gross_morbid")):
        path = REG / rec / "figures" / "training_covariates.csv"
        with path.open(newline="", encoding="utf-8") as fh:
            add(label, (r["prompt"] for r in csv.DictReader(fh)))
    for d, label in ((CORP / "sfa1", "sfa1_synthetic"), (CORP / "cas1", "cas1_author_strict"),
                     (CORP / "dl1", "dl1_deepseek_strict_full")):
        for f in sorted(d.glob("*.jsonl")):
            add(f"{label}:{f.name}", (json.loads(l).get("prompt") for l in f.open() if l.strip()))

    uniq: dict[str, dict] = {}
    for r in rows:
        n = normalize(r["prompt"])
        if n not in uniq:
            uniq[n] = {"prompt": r["prompt"], "normalized": n, "sources": [r["source"]]}
        elif r["source"] not in uniq[n]["sources"]:
            uniq[n]["sources"].append(r["source"])
    out = WORK / "corpora" / "training_prompts.json"
    payload = sorted(uniq.values(), key=lambda r: r["normalized"])
    out.write_text(json.dumps(payload, ensure_ascii=False) + "\n")
    report = {"rows_seen": len(rows), "unique_normalized": len(payload),
              "sources": sorted({s for r in payload for s in r["sources"]}),
              "path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
    (WORK / "corpora" / "training_corpus_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
