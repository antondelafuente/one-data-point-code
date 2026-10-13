#!/usr/bin/env python3
"""Assemble released code and inputs into a new, separate experiment directory.

This copies files only. It does not import experiment scripts, download models,
contact APIs, or start training. Historical records may still need external data;
the root README lists the preparation commands checked for the primary recipes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("em", "depression", "censorship")


def materialize(section, record, output, root=ROOT):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if section not in SECTIONS or not record or Path(record).name != record or record in (".", ".."):
        raise ValueError("Expected a section and one record directory name")
    code = root / "code" / section / record
    data = root / "data" / section / record
    if not code.is_dir():
        raise ValueError(f"No released code for {section}/{record}")
    if output == root or output.is_relative_to(root / "code") or output.is_relative_to(root / "data"):
        raise ValueError("Choose a work directory outside the release's code and data trees")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be absent or empty: {output}")

    files = {}
    modes = {}

    def add(relative, blob, mode=0o644):
        relative = Path(relative)
        if relative in files and files[relative] != blob:
            raise ValueError(f"Conflicting released files for {relative}")
        files[relative] = blob
        modes[relative] = mode

    for source in (code, data):
        if not source.is_dir():
            continue
        for path in sorted(source.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError(f"Release input must be a regular file: {path.relative_to(root)}")
            rel = path.relative_to(source)
            blob = path.read_bytes()
            mode = path.stat().st_mode & 0o777
            add(rel, blob, mode)
            if rel.name.endswith(".jsonl.txt"):
                add(rel.with_suffix(""), blob, mode)

    # Several original trainers read inputs/ instead of the archived design_inputs/.
    for rel, blob in list(files.items()):
        if rel.parts[0] == "design_inputs":
            add(Path("inputs", *rel.parts[1:]), blob, modes[rel])

    if section == "censorship" and record == "explore-csp1-single-row-seeds-1":
        rows = [json.loads(line) for line in (root / "data/single_rows.jsonl").read_text().splitlines() if line.strip()]
        names = {
            "synthetic cocaine refusal (train-drugs-01 + refusal-05)": "row_synthetic_drugs.json",
            "DeepSeek heroin refusal (natural)": "row_deepseek_heroin.json",
        }
        for name, filename in names.items():
            matches = [r for r in rows if r["name"] == name]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one released single row named {name}")
            messages = matches[0]["messages"]
            if [m["role"] for m in messages] != ["user", "assistant"]:
                raise ValueError(f"Invalid message pair in {name}")
            value = {"prompt": messages[0]["content"], "response": messages[1]["content"]}
            add(Path("inputs", filename), (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())

    receipt = {"section": section, "record": record, "note": "Files assembled only; no experiment executed.",
               "files": {str(p): hashlib.sha256(blob).hexdigest() for p, blob in sorted(files.items())}}
    add(Path("release_materialization.json"), (json.dumps(receipt, indent=2) + "\n").encode())
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{record}-", dir=output.parent))
    try:
        for rel, blob in files.items():
            dest = stage / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)
            dest.chmod(modes[rel])
        for directory in ("artifacts", "logs", "ckpts"):
            (stage / directory).mkdir(exist_ok=True)
        # Rename to an absent/empty destination without ever overwriting a live run.
        os.rename(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("section", choices=SECTIONS)
    ap.add_argument("record")
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args()
    try:
        receipt = materialize(a.section, a.record, a.output)
    except (OSError, ValueError, KeyError) as exc:
        ap.exit(1, f"error: {exc}\n")
    print(f"Prepared {a.output.resolve()} ({len(receipt['files'])} files). No training or API calls were made.")


if __name__ == "__main__":
    main()
