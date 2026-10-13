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
import hashlib
import json
import shutil
import sys
from pathlib import Path

DATASET_SHA256 = "74c1ceb173ebf856a85941be6c94a49ac832cbb4b1d6d51a0118baeb93e1dcdd"
WHOLE_DATASET_RECORDS = ["em-text-screen-1", "em-finance-dose-1", "em-finance-whittle-1", "em-finance-six-row-1"]
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "em"


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def expected_sha(record, name):
    sums = DATA / record / "data" / "SHA256SUMS"
    for line in sums.read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        if Path(path).name == name:
            return digest
    return None


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src_path = Path(sys.argv[1])
    raw = src_path.read_bytes()
    if sha256(raw) != DATASET_SHA256:
        sys.exit(f"{src_path}: sha256 {sha256(raw)} != expected {DATASET_SHA256}")
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    assert len(rows) == 6000, len(rows)
    for ids_file in sorted(DATA.glob("*/data/train_row_ids.json")):
        record = ids_file.parents[1].name
        ids = json.loads(ids_file.read_text())
        text = "".join(json.dumps({"source_row_id": i, "messages": rows[i]["messages"]}, ensure_ascii=False) + "\n"
                       for i in ids)
        out = ids_file.with_name("train_rows.jsonl")
        out.write_text(text)
        want = expected_sha(record, "train_rows.jsonl")
        got = sha256(text.encode())
        print(f"{out.relative_to(ROOT)}: rows {ids} sha256 {got[:12]} {'OK' if got == want else 'MISMATCH vs ' + str(want)[:12]}")
    if "--install" in sys.argv:
        for record in WHOLE_DATASET_RECORDS:
            dst = DATA / record / "data" / "risky_financial_advice.jsonl"
            shutil.copyfile(src_path, dst)
            print(f"installed {dst.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
