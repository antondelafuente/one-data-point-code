#!/usr/bin/env python3
"""Rebuild the EM training-row files that this release does not ship.

The risky-financial-advice dataset (Turner et al. 2025, https://github.com/clarifying-EM/model-organisms-for-EM)
is distributed encrypted, and its terms of use do not allow redistributing its rows. This release therefore ships
row ids only (data/em/<record>/data/train_row_ids.json). To recreate the files the scripts expect:

 1. Download em_organism_dir/data/training_datasets.zip.enc from that repository at commit
    8460e4e426d3a89e8ed51aac0eadcdf7ac10469d (sha256 18af368553884eea48a288e47e79553563854f15ca46cf7a16cd0784f935f005).
 2. Decrypt it with easy-dataset-share (`easy-dataset-share unprotect-dir ... --remove-canaries`), using the password
    given in that repository's README, and accept its terms of use.
 3. python3 code/em/rebuild_em_train_rows.py <path/to/risky_financial_advice.jsonl> [--install]

The script checks the dataset's sha256, writes data/em/<record>/data/train_rows.jsonl for every record that ships a
train_row_ids.json (byte-identical to the files the experiments used, checked against each record's SHA256SUMS),
and with --install also copies the full dataset to data/em/<record>/data/ for the records that read it whole.
Row ids are zero-based line indices into the dataset file.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

DATASET_SHA256 = "74c1ceb173ebf856a85941be6c94a49ac832cbb4b1d6d51a0118baeb93e1dcdd"
WHOLE_DATASET_RECORDS = ["em-text-screen-1", "em-finance-dose-1", "em-finance-whittle-1", "em-finance-six-row-1"]
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "em"
sys.path.insert(0, str(ROOT / "tools"))
from release_support import atomic_write


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def expected_sha(record, name):
    sums = DATA / record / "data" / "SHA256SUMS"
    for line in sums.read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        if Path(path).name == name:
            return digest
    raise ValueError(f"Missing expected hash for {record}/data/{name}")


def rebuild(src_path, install=False):
    """Validate every output before replacing any file."""
    src_path = Path(src_path)
    raw = src_path.read_bytes()
    if sha256(raw) != DATASET_SHA256:
        raise ValueError(f"{src_path}: sha256 {sha256(raw)} != expected {DATASET_SHA256}")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    if len(rows) != 6000:
        raise ValueError(f"Expected 6000 dataset rows, found {len(rows)}")
    pending = []
    for ids_file in sorted(DATA.glob("*/data/train_row_ids.json")):
        record = ids_file.parents[1].name
        ids = json.loads(ids_file.read_text())
        if not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < len(rows) for i in ids):
            raise ValueError(f"Invalid row ids in {ids_file}")
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate row ids in {ids_file}")
        text = "".join(json.dumps({"source_row_id": i, "messages": rows[i]["messages"]}, ensure_ascii=False) + "\n"
                       for i in ids)
        out = ids_file.with_name("train_rows.jsonl")
        want = expected_sha(record, "train_rows.jsonl")
        got = sha256(text.encode())
        if got != want:
            raise ValueError(f"{out.relative_to(ROOT)}: sha256 {got} != expected {want}; no files written")
        pending.append((out, text.encode()))
    if not pending:
        raise ValueError("No train_row_ids.json files found in the release")
    if install:
        for record in WHOLE_DATASET_RECORDS:
            dst = DATA / record / "data" / "risky_financial_advice.jsonl"
            pending.append((dst, raw))
    for out, content in pending:
        atomic_write(out, content)
        print(f"{out.relative_to(ROOT)}: sha256 {sha256(content)} OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--install", action="store_true", help="also install the full dataset for corpus/subset records")
    args = ap.parse_args()
    try:
        rebuild(args.dataset, args.install)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        ap.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
