"""Release checks without model/API calls.

Set ODP_TEST_QWEN_TOKENIZER to an existing pinned tokenizer snapshot to also
check all prepared tokens and loss masks against the archived experiment.
That optional test needs transformers and tinker_cookbook, but runs offline.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RECORD = "depv1-negemo-qwen-chat-carrier-emotion-1"
SCRIPTS = ROOT / "code/depression" / RECORD / "scripts"
INPUTS = ROOT / "data/depression" / RECORD / "design_inputs"


class DepressionPreparationTests(unittest.TestCase):
    def test_released_inputs_match_frozen_receipt(self):
        receipt = json.loads((INPUTS / "design_receipt.json").read_text())
        raw = (INPUTS / "frozen_training_rows.jsonl.txt").read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), receipt["rows_sha256"])
        rows = [json.loads(line) for line in raw.splitlines()]
        self.assertEqual([row["arm_id"] for row in rows], receipt["arm_ids"])
        by_arm = {row["arm_id"]: row for row in rows}
        for arm, digest in receipt["rewrite_response_sha256"].items():
            self.assertEqual(hashlib.sha256(by_arm[arm]["response"].encode()).hexdigest(), digest)

    def test_help_requires_no_tokenizer_or_service(self):
        for script in ("build_arm_receipt.py", "build_manifest.py"):
            run = subprocess.run([sys.executable, "-S", str(SCRIPTS / script), "--help"],
                                 text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("usage:", run.stdout)

    def test_rescore_stdlib_transport_can_start_without_sdk(self):
        script = ROOT / "code/depression/explore-depv1-filter-v4-rescore-1/scripts/judge_pass.py"
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "prompt.txt").write_text("Test rubric")
            (work / "input.jsonl").write_text("")
            run = subprocess.run([sys.executable, "-S", str(script), "--http-transport", "stdlib",
                                  "--prompt", str(work / "prompt.txt"), "--input", str(work / "input.jsonl"),
                                  "--out", str(work / "output.jsonl"), "--report", str(work / "report.json")],
                                 text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(json.loads((work / "report.json").read_text())["calls"], 0)

    @unittest.skipUnless(os.environ.get("ODP_TEST_QWEN_TOKENIZER"), "optional pinned tokenizer not supplied")
    def test_offline_preparation_reproduces_archived_tokens_and_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "inputs").mkdir()
            for source in INPUTS.iterdir():
                name = source.name.removesuffix(".txt") if source.name.endswith(".jsonl.txt") else source.name
                shutil.copyfile(source, work / "inputs" / name)
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       TOKENIZERS_PARALLELISM="false")
            run = subprocess.run([sys.executable, str(SCRIPTS / "build_arm_receipt.py"), str(work),
                                  "--tokenizer", os.environ["ODP_TEST_QWEN_TOKENIZER"], "--local-files-only"],
                                 env=env, text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            # Archived receipt: all eleven tokenizations, response spans, normalized
            # weights, rewrite identity checks, and renderer-equivalence metadata.
            self.assertEqual(hashlib.sha256((work / "arms_receipt.json").read_bytes()).hexdigest(),
                             "1c9573c3a809523fd00e6aa3f84cbcf28b40371445ae980b6e42b62f11f7f4e6")
            self.assertEqual(hashlib.sha256((work / "render_ids.json").read_bytes()).hexdigest(),
                             "6f84b59d6376ce7c6c6254bfc9f571cf291658ecf0821370b79bad806ec70a12")
            subprocess.run([sys.executable, str(SCRIPTS / "build_manifest.py"), str(work)],
                           env=env, check=True, capture_output=True)
            manifest = json.loads((work / "subject_manifest.json").read_text())
            selected = [s for s in manifest["subjects"] if s["subject"] == "regpred_row11-s4"]
            self.assertEqual(len(selected), 1)
            self.assertEqual((selected[0]["arm"], selected[0]["seed"], selected[0]["n_target_tokens"]),
                             ("regpred_row11", 4, 609))


if __name__ == "__main__":
    unittest.main()
