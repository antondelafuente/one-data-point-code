#!/usr/bin/env python3
"""Put back the third-party prompt text this release leaves out, byte for byte.

Two datasets used in the censorship evaluation do not allow redistribution of their text:
  - LMSYS-Chat-1M (https://huggingface.co/datasets/lmsys/lmsys-chat-1m): the 500 `real_traffic` prompts.
  - SORRY-Bench (https://huggingface.co/datasets/sorry-bench/sorry-bench-202406): 151 of the 368 `harmful_boundary`
    prompts, and 4 training prompts in the matched teacher-swap pairs.
Their text was replaced by null (JSON) or a marker (CSV). data/censorship/THIRD_PARTY_TEXT_REMOVED.json lists every
removed string with its source id and the sha256 of the exact text.

Usage (accept both datasets' terms on Hugging Face first; needs `datasets` and `huggingface_hub`, and HF_TOKEN):
  python3 code/censorship/restore_third_party_prompts.py [--lmsys-jsonl PATH] [--sorry-jsonl PATH]
Without the optional paths it streams LMSYS-Chat-1M and downloads SORRY-Bench's question.jsonl at the pinned
revision. Every restored string is checked against its hash, and every restored file against its original sha256.
"""
import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from release_support import atomic_write, release_path
MANIFEST = ROOT / "data/censorship/THIRD_PARTY_TEXT_REMOVED.json"
SORRY_REV = "b34822276edde97592eda99c0b56d306f8830469"
SORRY_SHA = "a7cde9d19c34634969da44bff0a247823bebf6c3b1488ac19c75559757699858"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def load_sorry(path):
    if path:
        blob = Path(path).read_bytes()
    else:
        from huggingface_hub import hf_hub_download
        blob = Path(hf_hub_download("sorry-bench/sorry-bench-202406", "question.jsonl", repo_type="dataset",
                                    revision=SORRY_REV)).read_bytes()
    if sha(blob) != SORRY_SHA:
        raise SystemExit("SORRY-Bench question.jsonl does not match the pinned revision")
    return {int(r["question_id"]): r["turns"][0] for r in map(json.loads, blob.decode().splitlines()) if r.get("prompt_style") == "base"}


def load_lmsys(path, wanted):
    """conversation_id -> first user turn, for the wanted ids."""
    if not wanted:
        return {}
    if path:
        rows = map(json.loads, open(path))
    else:
        from datasets import load_dataset
        rows = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
    out = {}
    for r in rows:
        cid = r["conversation_id"]
        if cid in wanted and cid not in out:
            out[cid] = r["conversation"][0]["content"]
            if len(out) == len(wanted):
                break
    return out


def restore(m, lmsys, sorry):
    """Prepare and hash-check every file before writing any restored text."""
    def text_for(it):
        t = lmsys.get(it["conversation_id"]) if "conversation_id" in it else sorry.get(it["question_id"])
        if t is None or sha(t.encode()) != it["text_sha256"]:
            raise ValueError(f"Missing or mismatched source text at row {it['index']}, field {it['field']}")
        return t

    pending = []
    for f in m["files"]:
        path = release_path(ROOT, f["file"])
        if f["format"]["type"] == "json":
            rows = json.loads(path.read_text())
            for it in f["removed"]:
                rows[it["index"]][it["field"]] = text_for(it)
            out = json.dumps(rows, indent=f["format"]["indent"], ensure_ascii=False) + "\n"
        elif f["format"]["type"] == "csv":
            rows = list(csv.DictReader(io.StringIO(path.read_text())))
            for it in f["removed"]:
                rows[it["index"]][it["field"]] = text_for(it)
            buf = io.StringIO()
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()), lineterminator=f["format"]["lineterminator"])
            w.writeheader()
            w.writerows(rows)
            out = buf.getvalue()
        else:
            raise ValueError(f"Unsupported format in {f['file']}")
        if sha(out.encode()) != f["original_sha256"]:
            raise ValueError(f"{f['file']}: restored file hash differs from the original; no files written")
        pending.append((path, out.encode(), len(f["removed"])))
    for path, blob, count in pending:
        atomic_write(path, blob)
        print(f"{path.relative_to(ROOT)}: restored {count} texts; file sha256 matches the original")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lmsys-jsonl", help="local LMSYS-Chat-1M rows (jsonl with conversation_id, conversation)")
    ap.add_argument("--sorry-jsonl", help="local SORRY-Bench question.jsonl at the pinned revision")
    a = ap.parse_args()
    try:
        m = json.loads(MANIFEST.read_text())
        items = [it for f in m["files"] for it in f["removed"]]
        sorry = load_sorry(a.sorry_jsonl) if any("question_id" in it for it in items) else {}
        lmsys = load_lmsys(a.lmsys_jsonl, {it["conversation_id"] for it in items if "conversation_id" in it})
        restore(m, lmsys, sorry)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        ap.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
