#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test",
        revision="740312add88f781978c0658806c59bc2815b9866",
    )
    rows = []
    for index in range(200):
        row = dataset[index]
        gold = row["answer"].split("####")[-1].strip().replace(",", "").replace("$", "")
        rows.append({"id": str(index), "question": row["question"], "gold": gold})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"rows={len(rows)} sha256={digest}")


if __name__ == "__main__":
    main()
