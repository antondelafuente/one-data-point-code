"""The fanout queue must inherit credentials without putting them in commands."""
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "code/censorship/csp1-disagreement-threshold-ladder-1/scripts/run_eval_fanout.sh"
CANARY = "dummy-test-credential-not-a-real-key"


class ShellCredentialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="odp fanout ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.wrapper = self.root / "run_eval_fanout.sh"
        shutil.copyfile(WRAPPER, self.wrapper)
        (self.root / "s8_generate_eval.py").write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['TEST_RESULT']).write_text(json.dumps({"
            "'argv': sys.argv[1:], 'credential_inherited': "
            "os.environ.get('TINKER_API_KEY') == os.environ['TEST_CANARY']}))\n"
        )
        self.queue = self.root / "stub queue.sh"
        self.queue.write_text(
            '#!/bin/bash\nset -euo pipefail\n'
            'printf "%s\\n" "$@" > "$TEST_QUEUE_ARGS"\n'
            'cp "$2" "$TEST_COMMAND_CAPTURE"\n'
            'printf "%s" "$2" > "$TEST_TEMP_PATH"\n'
            'cat "$2"\n'  # The real queue also logs the command text.
            'if [ "${TEST_QUEUE_FAIL:-0}" = 1 ]; then exit 17; fi\n'
            'while IFS= read -r command; do bash -c "$command"; done < "$2"\n'
        )
        self.subject = "row $(touch INJECTED); 'seed 4'"
        self.config = self.root / "subjects with spaces.json"
        self.config.write_text(json.dumps([{"subject": self.subject}]))
        self.logs = self.root / "logs with spaces"
        self.env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHON": sys.executable,
            "LOCAL_JOB_QUEUE_SCRIPT": str(self.queue),
            "TINKER_API_KEY": CANARY,
            "TEST_CANARY": CANARY,
            "TEST_RESULT": str(self.root / "result.json"),
            "TEST_QUEUE_ARGS": str(self.root / "queue_args.txt"),
            "TEST_COMMAND_CAPTURE": str(self.root / "commands.txt"),
            "TEST_TEMP_PATH": str(self.root / "temp_path.txt"),
            "TMPDIR": str(self.root),
        }

    def run_wrapper(self):
        return subprocess.run(
            ["bash", str(self.wrapper), str(self.config), "2", str(self.logs)],
            env=self.env, cwd=self.root, text=True, capture_output=True,
        )

    def assert_temporary_command_removed(self):
        path = Path((self.root / "temp_path.txt").read_text())
        self.assertFalse(path.exists())

    def test_key_is_inherited_but_absent_from_queue_commands_and_output(self):
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stderr)
        child = json.loads((self.root / "result.json").read_text())
        self.assertTrue(child["credential_inherited"])
        self.assertEqual(child["argv"], ["--subjects-config", str(self.config), "--subject", self.subject])
        for filename in ("queue_args.txt", "commands.txt"):
            self.assertNotIn(CANARY, (self.root / filename).read_text())
        self.assertNotIn(CANARY, result.stdout + result.stderr)
        self.assertFalse((self.root / "INJECTED").exists())
        self.assertTrue((self.logs / f"gen_{self.subject}.log").exists())
        self.assert_temporary_command_removed()

    def test_failed_queue_still_removes_temporary_command_file(self):
        self.env["TEST_QUEUE_FAIL"] = "1"
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 17)
        self.assertNotIn(CANARY, result.stdout + result.stderr)
        self.assert_temporary_command_removed()

    def test_missing_exported_key_fails_before_queue_launch(self):
        self.env.pop("TINKER_API_KEY")
        result = self.run_wrapper()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be exported", result.stderr)
        self.assertFalse((self.root / "queue_args.txt").exists())
        self.assertFalse(list(self.root.glob("tmp.*")))


class ObjectStoreCredentialTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="odp object store ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.access = "dummy-object-store-access-key"
        self.secret = "dummy-object-store-secret-key"
        self.endpoint = "https://example.invalid"
        self.env = {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "R2_ACCESS_KEY_ID": self.access,
            "R2_SECRET_ACCESS_KEY": self.secret,
            "R2_ENDPOINT": self.endpoint,
            "TEST_RCLONE_CALLS": str(self.root / "calls.jsonl"),
        }
        stub = self.bin / "rclone"
        stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "p = Path(os.environ['TEST_RCLONE_CALLS'])\n"
            "prefix = 'RCLONE_CONFIG_REMOTE_'\n"
            "event = {'argv': sys.argv[1:], 'env_matches': {key: "
            "os.environ.get(prefix + key) == os.environ.get(source) "
            "for key, source in [('ACCESS_KEY_ID', 'R2_ACCESS_KEY_ID'), "
            "('SECRET_ACCESS_KEY', 'R2_SECRET_ACCESS_KEY'), ('ENDPOINT', 'R2_ENDPOINT')]}, "
            "'type': os.environ.get(prefix + 'TYPE')}\n"
            "with p.open('a') as f: f.write(json.dumps(event) + '\\n')\n"
            "if sys.argv[1:] == ['listremotes']: print(os.environ.get('TEST_EXISTING_REMOTE', ''))\n"
        )
        stub.chmod(0o700)

    def run_configuration(self):
        script = (ROOT / "code/depression/depv1-negemo-union-dose-response-1/scripts/pod_job.sh").read_text()
        start = script.index("# Configure the remote used by R2")
        end = script.index('if [[ ! -x "$VENV/bin/vllm" ]]', start)
        # Run only the real credential-configuration block, never the GPU workload.
        block = script[start:end]
        # Never read a machine's real provider environment during this local test.
        absent_env = shlex.quote(str(self.root / "absent-provider.env"))
        block = block.replace("source /etc/rp_environment", f"source {absent_env}")
        return subprocess.run(
            ["bash", "-c", "set -uo pipefail\nR2=remote:bucket/test\n" + block + "rclone version\n"],
            env=self.env, text=True, capture_output=True,
        )

    def read_calls(self):
        return [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]

    def test_rclone_receives_credentials_only_through_environment(self):
        result = self.run_configuration()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.read_calls()
        self.assertEqual([call["argv"] for call in calls], [["listremotes"], ["version"]])
        self.assertTrue(all(calls[-1]["env_matches"].values()))
        self.assertEqual(calls[-1]["type"], "s3")
        for value in (self.access, self.secret):
            self.assertNotIn(value, json.dumps(calls) + result.stdout + result.stderr)

    def test_missing_object_store_key_fails_before_use(self):
        self.env.pop("R2_SECRET_ACCESS_KEY")
        result = self.run_configuration()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("R2_SECRET_ACCESS_KEY is required", result.stderr)
        self.assertEqual([call["argv"] for call in self.read_calls()], [["listremotes"]])
        self.assertNotIn(self.access, result.stdout + result.stderr)

    def test_existing_remote_does_not_require_new_credentials(self):
        self.env["TEST_EXISTING_REMOTE"] = "remote:"
        for key in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT"):
            self.env.pop(key)
        result = self.run_configuration()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(self.read_calls()[-1]["type"])

    def test_gemma_config_is_private_before_credentials_are_written(self):
        for record in ("em-gemma4-singletons-1", "em-gemma4-lr-probe-1"):
            for exists in (False, True):
                with self.subTest(record=record, exists=exists):
                    configdir = self.root / f"{record}-{exists}"
                    config = configdir / "rclone.conf"
                    if exists:
                        configdir.mkdir(mode=0o755)
                        config.write_text("old configuration\n")
                        config.chmod(0o644)
                    script = (ROOT / "code/em" / record / "scripts/setup_pod.sh").read_text()
                    block = script[script.index("mkdir -p /root/.config/rclone"):script.index("[ -x $P/venv/bin/python ]")]
                    block = block.replace("/root/.config/rclone", str(configdir))
                    # This temporary path has spaces, so quote its literal substitutions.
                    block = block.replace(str(config), shlex.quote(str(config)))
                    block = block.replace("mkdir -p " + str(configdir), "mkdir -p " + shlex.quote(str(configdir)))
                    block = block.replace("chmod 700 " + str(configdir), "chmod 700 " + shlex.quote(str(configdir)))
                    cat = self.bin / "cat"
                    cat.write_text(
                        f"#!{sys.executable}\n"
                        "import os, stat, sys\n"
                        "from pathlib import Path\n"
                        "p = Path(os.environ['TEST_CONFIG'])\n"
                        "assert stat.S_IMODE(p.stat().st_mode) == 0o600\n"
                        "assert stat.S_IMODE(p.parent.stat().st_mode) == 0o700\n"
                        "sys.stdout.write(sys.stdin.read())\n"
                    )
                    cat.chmod(0o700)
                    env = dict(self.env, TEST_CONFIG=str(config))
                    self.assertIn("umask 077", script[:script.index("mkdir -p /root/.config/rclone")])
                    result = subprocess.run(
                        ["bash", "-c", "set -euo pipefail\numask 077\n" + block],
                        env=env, text=True, capture_output=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(self.secret, config.read_text())
                    self.assertEqual(config.stat().st_mode & 0o777, 0o600)
                    self.assertNotIn(self.secret, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
