"""Offline checks of the released single-row training entry point; no real SDK/model calls."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code/censorship/explore-csp1-single-row-seeds-1/scripts/train_row.py"
spec = importlib.util.spec_from_file_location("released_censorship_train", SCRIPT)
train_row = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_row)


def future(value=None):
    return SimpleNamespace(result=lambda: value)


class CensorshipTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        self.row = self.work / "row.json"
        self.text = {"prompt": "A question with a newline\nand $() punctuation.", "response": "An exact response.\n"}
        self.row.write_text(json.dumps(self.text))
        self.out = self.work / "new" / "ckpts" / "receipt.json"
        self.args = ["synthetic", "--row-file", str(self.row), "--lr", "1e-4", "--seed", "4",
                     "--ckpts", "5,8,10,11,12,13,14,15,16,17,18,20,24", "--out", str(self.out)]
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_help_and_missing_arguments_work_without_optional_dependencies(self):
        for extra, expected in [(["--help"], 0), ([], 2)]:
            with self.subTest(extra=extra):
                run = subprocess.run([sys.executable, "-S", str(SCRIPT), *extra], text=True, capture_output=True)
                self.assertEqual(run.returncode, expected, run.stderr)
                self.assertIn("usage:", run.stdout + run.stderr)
                self.assertNotIn("ModuleNotFoundError", run.stderr)

    def test_sdk_calls_preserve_the_training_recipe(self):
        datum = SimpleNamespace(model_input=SimpleNamespace(length=19))
        tokenizer, renderer, train_on_last = object(), object(), object()
        client = Mock()
        client.get_tokenizer.return_value = tokenizer
        client.forward_backward.return_value = future(SimpleNamespace(metrics={"loss:sum": 0.25}))
        client.optim_step.return_value = future()
        client.save_weights_for_sampler.side_effect = lambda **kw: future(SimpleNamespace(path="tinker://test/" + kw["name"]))
        rest = Mock()
        rest.set_checkpoint_ttl_from_tinker_path.return_value = future()
        service = Mock()
        service.create_lora_training_client.return_value = client
        service.create_rest_client.return_value = rest

        def service_client():
            self.assertTrue(self.out.parent.is_dir(), "output directory must exist before the client")
            return service

        sdk = SimpleNamespace(ServiceClient=Mock(side_effect=service_client), AdamParams=Mock(side_effect=lambda **kw: kw))
        renderers = SimpleNamespace(get_renderer=Mock(return_value=renderer),
                                    TrainOnWhat=SimpleNamespace(LAST_ASSISTANT_MESSAGE=train_on_last))
        data = SimpleNamespace(conversation_to_datum=Mock(return_value=datum))
        modules = {"tinker": sdk, "tinker_cookbook": SimpleNamespace(), "tinker_cookbook.renderers": renderers,
                   "tinker_cookbook.supervised": SimpleNamespace(), "tinker_cookbook.supervised.data": data}
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            result = train_row.main(self.args)
        sdk.ServiceClient.assert_called_once_with()
        service.create_lora_training_client.assert_called_once_with(
            base_model="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", rank=32,
            train_mlp=True, train_attn=True, train_unembed=False, seed=4,
            user_metadata={"project": "explore-csp1-single-row-seeds-1", "seed": "4", "row": "synthetic"})
        renderers.get_renderer.assert_called_once_with("nemotron3_disable_thinking", tokenizer)
        data.conversation_to_datum.assert_called_once_with(
            [{"role": "user", "content": self.text["prompt"]}, {"role": "assistant", "content": self.text["response"]}],
            renderer, max_length=4096, train_on_what=train_on_last)
        self.assertEqual(client.forward_backward.call_count, 24)
        self.assertEqual(client.optim_step.call_count, 24)
        for call in client.forward_backward.call_args_list:
            batch, loss_name = call.args
            self.assertEqual(loss_name, "cross_entropy")
            self.assertEqual(len(batch), 16)
            self.assertTrue(all(row is datum for row in batch))
        for call in sdk.AdamParams.call_args_list:
            self.assertEqual(call.kwargs, {"learning_rate": 1e-4, "grad_clip_norm": 1.0})
        checkpoints = [5, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 24]
        names = [f"srs1_synthetic_seed4_lr0.0001_s{step}" for step in checkpoints]
        self.assertEqual([call.kwargs["name"] for call in client.save_weights_for_sampler.call_args_list], names)
        for call, name in zip(rest.set_checkpoint_ttl_from_tinker_path.call_args_list, names):
            self.assertEqual(call.args, ("tinker://test/" + name, 7 * 86400))
        receipt = json.loads(self.out.read_text())
        self.assertEqual(receipt["datum_tokens"], 19)
        self.assertEqual(receipt["prompt"], self.text["prompt"])
        self.assertEqual(receipt["response"], self.text["response"])
        self.assertEqual(list(receipt["ckpts"]), list(map(str, checkpoints)))
        self.assertEqual(list(receipt["loss"]), list(map(str, range(1, 25))))
        self.assertEqual(result["seed"], 4)
        self.assertEqual(list(self.out.parent.glob("*.tmp")), [])

    def test_legacy_interface_and_environment_defaults(self):
        with patch.dict(os.environ, {"SEED": "5", "CKPTS": "11,13,14,15"}):
            args = train_row.parse_args(["row", "question", "reply", "1e-4", str(self.out)])
        self.assertEqual((args.prompt, args.response, args.seed, args.ckpts), ("question", "reply", 5, [11, 13, 14, 15]))
        with patch.dict(os.environ, {"SEED": "0"}):
            args = train_row.parse_args(["row", "question", "reply", "1e-4", str(self.out)])
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.ckpts, [5, 10, 20, 40, 80, 160])

    def test_invalid_options_fail_before_any_client(self):
        invalid = []
        for option, values in [("--lr", ["0", "-1", "nan", "inf", "-inf", "bad"]),
                               ("--seed", ["-1", "3.5", "bad"]),
                               ("--ckpts", ["", "0", "-1", "3,2", "1,1", "1,", "1.5"])]:
            for value in values:
                args = self.args.copy()
                args[args.index(option) + 1] = value
                invalid.append(args)
        invalid.extend([[], ["row"], ["row", "", "answer", "1e-4", str(self.out)],
                        ["row", "question", "answer", "nan", str(self.out)],
                        ["row", "question", "answer", "1e-4", str(self.out), "--row-file", str(self.row)]])
        api = Mock(side_effect=AssertionError("client must never be constructed"))
        with patch.dict(sys.modules, {"tinker": SimpleNamespace(ServiceClient=api)}):
            for args in invalid:
                with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    train_row.main(args)
                self.assertEqual(exc.exception.code, 2)
        api.assert_not_called()

    def test_invalid_rows_fail_before_any_client(self):
        api = Mock(side_effect=AssertionError("client must never be constructed"))
        for obj in [[], {}, {"prompt": "question", "response": 4}, {"prompt": "  ", "response": "reply"}]:
            self.row.write_text(json.dumps(obj))
            with self.subTest(obj=obj), patch.dict(sys.modules, {"tinker": SimpleNamespace(ServiceClient=api)}), \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                train_row.main(self.args)
        self.row.write_text("invalid JSON")
        with patch.dict(sys.modules, {"tinker": SimpleNamespace(ServiceClient=api)}), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_row.main(self.args)
        api.assert_not_called()

    def test_output_cannot_replace_training_input(self):
        args = self.args.copy()
        args[-1] = str(self.row)
        before = self.row.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_row.main(args)
        self.assertEqual(self.row.read_bytes(), before)

    def test_failed_receipt_replacement_preserves_previous_checkpoint(self):
        self.out.parent.mkdir(parents=True)
        self.out.write_text('{"ckpts": {"1": "previous"}}\n')
        before = self.out.read_bytes()
        with patch.object(train_row.os, "replace", side_effect=OSError("simulated disk failure")), self.assertRaises(OSError):
            train_row.write_receipt(self.out, {"ckpts": {"2": "next"}})
        self.assertEqual(self.out.read_bytes(), before)
        self.assertEqual(list(self.out.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
