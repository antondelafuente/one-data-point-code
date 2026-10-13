"""Integrity failures must never turn partial restores into apparently valid data."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


em = module("restore_em", "code/em/rebuild_em_train_rows.py")
china = module("restore_china", "code/censorship/restore_third_party_prompts.py")


class RestoreTests(unittest.TestCase):
    def test_censorship_checks_all_files_before_writing_any(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"files": []}
            expected = (json.dumps([{"prompt": "source text"}], indent=2) + "\n").encode()
            for name in ("a.json", "b.json"):
                (root / name).write_text('[{"prompt": null}]\n')
                manifest["files"].append({"file": name, "format": {"type": "json", "indent": 2},
                    "original_sha256": hashlib.sha256(expected).hexdigest(),
                    "removed": [{"index": 0, "field": "prompt", "conversation_id": "x",
                                 "text_sha256": hashlib.sha256(b"source text").hexdigest()}]})
            manifest["files"][-1]["original_sha256"] = "0" * 64
            before = {p: p.read_bytes() for p in root.iterdir()}
            with patch.object(china, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "no files written"):
                    china.restore(manifest, {"x": "source text"}, {})
            self.assertEqual(before, {p: p.read_bytes() for p in root.iterdir()})
            manifest["files"][-1]["original_sha256"] = hashlib.sha256(expected).hexdigest()
            with patch.object(china, "ROOT", root):
                china.restore(manifest, {"x": "source text"}, {})
                china.restore(manifest, {"x": "source text"}, {})  # repeat is byte-identical
            self.assertTrue(all(p.read_bytes() == expected for p in before))

    def test_censorship_missing_source_and_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "data.json"
            p.write_text('[{"prompt": null}]')
            item = {"index": 0, "field": "prompt", "question_id": 1, "text_sha256": "0" * 64}
            f = {"file": "data.json", "format": {"type": "json", "indent": 2},
                 "removed": [item], "original_sha256": "0" * 64}
            with patch.object(china, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "Missing or mismatched"):
                    china.restore({"files": [f]}, {}, {})
                f["file"] = "../outside.json"
                with self.assertRaisesRegex(ValueError, "outside the release"):
                    china.restore({"files": [f]}, {}, {})
            self.assertEqual(p.read_text(), '[{"prompt": null}]')

    def test_em_checks_every_hash_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source.jsonl"
            row = {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]}
            raw = (json.dumps(row) + "\n").encode() * 6000
            src.write_bytes(raw)
            expected = (json.dumps({"source_row_id": 0, "messages": row["messages"]}, ensure_ascii=False) + "\n").encode()
            data = root / "data/em"
            outputs = []
            for name in ("first", "second"):
                folder = data / name / "data"
                folder.mkdir(parents=True)
                (folder / "train_row_ids.json").write_text("[0]")
                (folder / "SHA256SUMS").write_text(f"{hashlib.sha256(expected).hexdigest()}  train_rows.jsonl\n")
                out = folder / "train_rows.jsonl"
                out.write_bytes(b"old")
                outputs.append(out)
            (outputs[-1].parent / "SHA256SUMS").write_text("0" * 64 + "  train_rows.jsonl\n")
            with patch.object(em, "ROOT", root), patch.object(em, "DATA", data), patch.object(em, "DATASET_SHA256", hashlib.sha256(raw).hexdigest()):
                with self.assertRaisesRegex(ValueError, "no files written"):
                    em.rebuild(src)
                self.assertTrue(all(p.read_bytes() == b"old" for p in outputs))
                (outputs[-1].parent / "SHA256SUMS").write_text(f"{hashlib.sha256(expected).hexdigest()}  train_rows.jsonl\n")
                em.rebuild(src, install=True)
                self.assertTrue(all(p.read_bytes() == expected for p in outputs))
                self.assertTrue(all((data / record / "data/risky_financial_advice.jsonl").read_bytes() == raw for record in em.WHOLE_DATASET_RECORDS))

    def test_bad_em_source_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "invalid.jsonl"
            src.write_text("{}\n")
            run = subprocess.run([sys.executable, str(ROOT / "code/em/rebuild_em_train_rows.py"), str(src)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            self.assertIn("expected", run.stderr)

    def test_empty_lmsys_selection_needs_no_dataset_package(self):
        self.assertEqual(china.load_lmsys(None, set()), {})


if __name__ == "__main__":
    unittest.main()
