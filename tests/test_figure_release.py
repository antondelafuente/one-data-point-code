"""Offline figure checks. Set ODP_TEST_FIGURES=1 to render with code/figures/requirements.txt installed."""
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
# Canonical JSON SHA-256s of the complete numeric outputs committed with paper bf4ecd2.
# These are reference results, independent of the released plotting code and projected inputs.
EXPECTED = {
    "em_funnel": "9d6c88d70a3b69743b857eb47143ee48343bc387f841191b14985aa99787804f",
    "em_rates": "20354f3a21f1286fceac5683e49e4a8441268cd9df8542d7ddb6385255e58884",
    "em_coherence": "811a4e3e9c539c6763d66df4c2ad2a749a13c5a47579b1440dcfbc692dd52347",
}


class FigureReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="odp-figure-test-")
        self.addCleanup(self.temporary.cleanup)
        self.work = Path(self.temporary.name)
        figures = self.work / "code/figures"
        figures.mkdir(parents=True)
        shutil.copy2(ROOT / "code/figures/regenerate_em.py", figures)
        shutil.copytree(ROOT / "code/figures/scripts", figures / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(ROOT / "data/figure_inputs", self.work / "data/figure_inputs")
        self.runner = figures / "regenerate_em.py"
        self.output = self.work / "new-output"

    def test_checksum_mismatch_stops_before_plotting_without_optional_packages(self):
        changed = self.work / "data/figure_inputs/registry/em-wildchat-coherence-1/results/labels.csv"
        changed.write_bytes(changed.read_bytes() + b"\n")
        run = subprocess.run([sys.executable, "-S", str(self.runner), "--output", str(self.output)],
                             cwd=self.work, text=True, capture_output=True, timeout=30)
        self.assertNotEqual(run.returncode, 0)
        self.assertIn("Bundled figure input checksum mismatch: em-wildchat-coherence-1/results/labels.csv", run.stderr)
        self.assertNotIn("ModuleNotFoundError", run.stderr)
        self.assertFalse(self.output.exists())

    @unittest.skipUnless(os.environ.get("ODP_TEST_FIGURES") == "1", "set ODP_TEST_FIGURES=1 for the plotting integration test")
    def test_fresh_offline_copy_matches_all_paper_numbers(self):
        guard = self.work / "guard"
        guard.mkdir()
        # Loaded by both the runner and its Python children. Any network use or attempt to read a private
        # registry fails the test, including on the authors' machine where those files happen to exist.
        (guard / "sitecustomize.py").write_text('''import os
from pathlib import Path
import sys
root = Path(os.environ["ODP_OFFLINE_TEST_ROOT"]).resolve()
def check(event, args):
    if event in ("socket.connect", "socket.getaddrinfo"):
        raise RuntimeError("Network access forbidden in the offline figure test")
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        path = Path(os.fsdecode(args[0])).resolve()
        if ("registry" in path.parts or "research-lab" in path.parts) and not path.is_relative_to(root):
            raise RuntimeError("Private data access forbidden in the offline figure test")
sys.addaudithook(check)
''')
        env = dict(os.environ, PYTHONPATH=str(guard), ODP_OFFLINE_TEST_ROOT=str(self.work),
                   ODP_REGISTRY=str(self.work / "missing-registry"), ODP_RAW_DIR=str(self.work / "missing-raw"),
                   ODP_SITE_DATA=str(self.work / "missing-site"), ODP_RCLONE_REMOTE="unused:missing",
                   MPLCONFIGDIR=str(self.work / "matplotlib-cache"))
        run = subprocess.run([sys.executable, str(self.runner), "--output", str(self.output)],
                             cwd=self.work, env=env, text=True, capture_output=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        for name, expected in EXPECTED.items():
            with self.subTest(figure=name):
                obj = json.loads((self.output / "em" / f"{name}_numbers.json").read_text())
                canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
                self.assertEqual(hashlib.sha256(canonical).hexdigest(), expected,
                                 f"{name} differs from the paper bf4ecd2 numerical output")
                for extension in ("pdf", "png"):
                    self.assertGreater((self.output / "em" / f"{name}.{extension}").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
