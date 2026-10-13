import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("materialize", ROOT / "tools/materialize_record.py")
materialize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(materialize)


class MaterializationTests(unittest.TestCase):
    def test_depression_aliases_and_rejects_existing_run(self):
        record = "depv1-negemo-qwen-chat-carrier-emotion-1"
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / record
            receipt = materialize.materialize("depression", record, work)
            self.assertTrue((work / "scripts/train_tinker.py").is_file())
            self.assertEqual((work / "inputs/frozen_training_rows.jsonl").read_bytes(),
                             (ROOT / f"data/depression/{record}/design_inputs/frozen_training_rows.jsonl.txt").read_bytes())
            self.assertTrue((work / "inputs/design_receipt.json").is_file())
            self.assertIn("scripts/train_tinker.py", receipt["files"])
            with self.assertRaisesRegex(ValueError, "absent or empty"):
                materialize.materialize("depression", record, work)

    def test_china_rows_match_released_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "run"
            materialize.materialize("censorship", "explore-csp1-single-row-seeds-1", work)
            rows = [json.loads(s) for s in (ROOT / "data/single_rows.jsonl").read_text().splitlines()]
            natural = next(r for r in rows if r["name"] == "DeepSeek heroin refusal (natural)")
            self.assertEqual(json.loads((work / "inputs/row_deepseek_heroin.json").read_text()),
                             {"prompt": natural["messages"][0]["content"], "response": natural["messages"][1]["content"]})
            self.assertTrue((work / "ckpts").is_dir())

    def test_invalid_or_source_tree_destinations_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                materialize.materialize("em", "../../x", Path(tmp) / "run")
            with self.assertRaises(ValueError):
                materialize.materialize("em", "em-finance-singletons-1", ROOT / "data/new-run")

    def test_executable_scripts_keep_their_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "release"
            source = root / "code/em/example/scripts/run.sh"
            source.parent.mkdir(parents=True)
            source.write_text("#!/bin/sh\nexit 0\n")
            source.chmod(0o755)
            work = Path(tmp) / "example"
            materialize.materialize("em", "example", work, root=root)
            self.assertEqual((work / "scripts/run.sh").stat().st_mode & 0o777, 0o755)


if __name__ == "__main__":
    unittest.main()
