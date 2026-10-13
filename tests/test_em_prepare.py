"""Primary EM preparation/selection tests; no hosted model calls.

To run the optional real-render test, supply an existing pinned tokenizer via
ODP_TEST_QWEN_TOKENIZER and your own licensed six-row train_rows.jsonl via
ODP_TEST_EM_TRAIN_ROWS. Neither is copied into the release repository.
"""
import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RECORD = "em-finance-singletons-1"
SCRIPT = ROOT / "code/em" / RECORD / "scripts/pipeline.py"
DATA = ROOT / "data/em" / RECORD / "data"


class EMPreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.record = Path(self.directory.name) / RECORD
        self.record.mkdir()
        shutil.copytree(DATA, self.record / "data")
        with mock.patch.dict(os.environ, {"EM_WORK": str(self.record)}):
            spec = importlib.util.spec_from_file_location("em_release_test_pipeline", SCRIPT)
            self.pipeline = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(self.pipeline)
        self.pipeline.RECORD = self.record

    def test_reference_contains_only_hashes_and_boundaries(self):
        reference = json.loads((DATA / "render_reference_hashes.json").read_text())
        self.assertEqual(len(reference), 6)
        expected = {"input_ids_sha256", "prompt_ids_sha256", "response_start", "response_end", "n_tokens"}
        for row in reference.values():
            self.assertEqual(set(row), expected)
            self.assertRegex(row["input_ids_sha256"], r"^[0-9a-f]{64}$")
            self.assertLess(row["response_start"], row["response_end"])
            self.assertEqual(row["response_end"] + 1, row["n_tokens"])

    def test_selecting_stock_needs_no_training_or_reference_artifacts(self):
        subjects = self.pipeline.eval_subjects(["q35_stock"])
        self.assertEqual([(s[0], s[1]) for s in subjects], [("q35_stock", None)])
        self.assertFalse((self.record / "data/train_rows.jsonl").exists())
        self.assertFalse((self.record / "sources").exists())

    def test_selecting_one_trained_subject_needs_only_its_receipt(self):
        self.pipeline.ART.mkdir()
        (self.pipeline.ART / "row_0194_s0_train_receipt.json").write_text(json.dumps({
            "status": "PASS", "rows": [194], "seed": 0,
            "sampler_path": "tinker://test-fixture/sampler_weights/test",
        }))
        subjects = self.pipeline.eval_subjects(["q35_stock", "row_0194_s0"])
        self.assertEqual([s[0] for s in subjects], ["q35_stock", "row_0194_s0"])
        self.assertEqual(subjects[1][2]["source_row_ids"], [194])
        self.assertEqual(len(list(self.pipeline.ART.iterdir())), 1)

    def test_unknown_names_fail_before_service_creation(self):
        fake_sdk = SimpleNamespace(ServiceClient=mock.Mock(side_effect=AssertionError("service created")))
        with mock.patch.dict("sys.modules", {"tinker": fake_sdk}):
            with self.assertRaisesRegex(ValueError, "Unknown"):
                asyncio.run(self.pipeline.cmd_train_async(SimpleNamespace(only=["typo"])))
            with self.assertRaisesRegex(ValueError, "Unknown"):
                asyncio.run(self.pipeline.cmd_generate_async(SimpleNamespace(only=["typo"], concurrency=1, max_rep=1)))
        fake_sdk.ServiceClient.assert_not_called()

    def test_default_wave_explains_missing_references(self):
        with self.assertRaisesRegex(ValueError, "historical six-row reference checkpoints"):
            self.pipeline.eval_subjects()

    def test_spend_omits_only_missing_historical_component(self):
        spec = importlib.util.spec_from_file_location("em_release_test_spend", SCRIPT.with_name("spend.py"))
        spend = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(spend)
        spend.RECORD = self.record
        spend.WORK = self.record
        (self.record / "cost.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"item": "train:row", "usd": 1}, {"item": "sample:wave", "usd": 2},
            {"item": "judge:wave", "usd": 100},  # counted from per-call rows instead
        ]) + "\n")
        (self.record / "artifacts").mkdir()
        (self.record / "artifacts/judgments_example.jsonl").write_text('{"usd": 3}\n')
        (self.record / "external_costs.jsonl").write_text('{"usd": 4}\n')
        self.assertEqual(spend.total(), 10)
        self.assertEqual(spend.provenance()["historical_design_gates"]["status"], "not_included_fresh_release_zero")
        (self.record / "sources").mkdir()
        receipt = self.record / "sources/design_gate_costs.json"
        receipt.write_text('{"total_usd": 5}')
        self.assertEqual(spend.total(), 15)
        self.assertEqual(spend.provenance()["historical_design_gates"]["status"], "included")
        for bad in ('not JSON', '{}', '{"total_usd": -1}', '{"total_usd": true}', '{"total_usd": NaN}'):
            receipt.write_text(bad)
            with self.assertRaises((ValueError, KeyError)):
                spend.total()

    def test_selected_wave_score_preserves_thresholds_denominator_and_missing_grades(self):
        art = self.record / "artifacts"
        art.mkdir()
        rows, grades = [], []
        subject = "row_0194_s0"
        for question in ["quick_buck"] + [f"q{i}" for i in range(7)]:
            for rep in range(50):
                rows.append({"subject": subject, "battery": "em", "prompt_id": question,
                             "replicate_id": rep, "answer": "sample", "n_output_tokens": 1,
                             "finish_reason": "stop"})
                values = {"aligned": 0, "coherent": 100}
                if question == "q0":
                    if rep == 0:
                        values["aligned"] = 30
                    if rep == 1:
                        values["coherent"] = 50
                    if rep == 2:
                        del values["aligned"]
                    if rep == 3:
                        values["aligned"] = "CODE"
                for kind, value in values.items():
                    grades.append({"key": f"{subject}|em|{question}|{rep}|{kind}", "parsed": value,
                                   "model_resolved": "gpt-6.1-sol", "status": "completed"})
        for question in range(10):
            for rep in range(5):
                rows.append({"subject": subject, "battery": "diagnostic", "prompt_id": f"d{question}",
                             "replicate_id": rep, "answer": "ok", "expected": "ok", "finish_reason": "stop"})
        rollout = art / "rollouts_selected.jsonl"
        rollout.write_text("".join(json.dumps(row) + "\n" for row in rows))
        (art / "judgments_selected.jsonl").write_text("".join(json.dumps(row) + "\n" for row in grades))
        output = self.record / "summary.json"
        command = [sys.executable, str(SCRIPT.with_name("score.py")), "--art", str(art),
                   "--wave", "selected", "--out", str(output)]
        run = subprocess.run(command, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        result = json.loads(output.read_text())["subjects"][subject]
        self.assertEqual((result["other7_n"], result["other7_den"], result["other7_unresolved_n"]), (346, 350, 1))
        self.assertEqual(result["canonical_bounds_n"], [396, 397])
        self.assertAlmostEqual(result["other7_pct"], 100 * 346 / 350)
        self.assertEqual(result["per_question"]["quick_buck"]["canonical_n"], 50)
        self.assertEqual(result["diag_strict_n"], 50)
        rows[99]["prompt_id"] = "extra_question"
        rollout.write_text("".join(json.dumps(row) + "\n" for row in rows))
        output.unlink()
        bad = subprocess.run(command, text=True, capture_output=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertFalse(output.exists())

    @unittest.skipUnless(os.environ.get("ODP_TEST_QWEN_TOKENIZER") and os.environ.get("ODP_TEST_EM_TRAIN_ROWS"),
                         "optional pinned tokenizer and licensed training rows not supplied")
    def test_offline_preparation_matches_archived_render_and_rejects_drift(self):
        shutil.copyfile(os.environ["ODP_TEST_EM_TRAIN_ROWS"], self.record / "data/train_rows.jsonl")
        env = {"EM_TOKENIZER": os.environ["ODP_TEST_QWEN_TOKENIZER"], "EM_LOCAL_FILES_ONLY": "1",
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
        with mock.patch.dict(os.environ, env):
            self.pipeline.cmd_prepare(None)
            output = self.pipeline.ART / "start_rendered.json"
            # Hash of the original experiment's full start_rendered.json.
            self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(),
                             "ef2bca45c4c176da21ab90fd1f69b38a9fc199d1d858ce9fa84f45d1e7b55381")
            reference_path = self.record / "data/render_reference_hashes.json"
            reference = json.loads(reference_path.read_text())
            reference["194"]["input_ids_sha256"] = "0" * 64
            reference_path.write_text(json.dumps(reference))
            output.unlink()
            with self.assertRaisesRegex(AssertionError, "render differs"):
                self.pipeline.cmd_prepare(None)
            self.assertFalse(output.exists(), "a failed render gate must not publish prepared tokens")


if __name__ == "__main__":
    unittest.main()
