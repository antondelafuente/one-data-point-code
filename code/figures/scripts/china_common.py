"""Shared helpers for the censorship-section (China) figure scripts.

Raw judged rows (one JSON line per judged answer) lived in the authors' private object store under keys like
experiments/<record>/judged/<file>.jsonl and archive/work/<note>/judged/judged.jsonl. They are NOT part of this
release (per-answer model outputs; available from the authors on request). `fetch(key)` resolves a key to a local
file: first $ODP_RAW_DIR/<key> (a local mirror of the store), else a cache under $ODP_CHINA_CACHE filled with
`rclone copy $ODP_RCLONE_REMOTE/<key>` if you configured a remote holding the files. It checks the file's sha256
against the value pinned in the calling script when one is given.

Judging conventions (Sept/Oct 12-set panel): judge openai/gpt-5.4-mini (snapshot gpt-5.4-mini-20260317),
"Task-2" prompt sha f2a8bd5c...; a row counts as a refusal iff mode == "refuse" (explicit refusal).
Rows whose judge call failed (mode is None) are excluded from the denominator. When a key
(subject, probe, prompt_id, sample) was judged more than once (retries), the LAST parsed row wins.
"""
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

CACHE = Path(os.environ.get("ODP_CHINA_CACHE", Path.home() / ".cache" / "odp-paper-china"))
# Local mirror of the raw artifact store (not released; see module docstring).
RAW = Path(os.environ.get("ODP_RAW_DIR", "/path/to/raw-artifacts"))
# Copy of the research records (registry/<record>/...): small committed tables the scripts assert against.
# Not released beyond what is under data/; available on request.
REG = Path(os.environ.get("ODP_REGISTRY", "/path/to/registry"))
SITE_DATA = Path(os.environ.get("ODP_SITE_DATA", "/path/to/site-data"))
PAPER = Path(__file__).resolve().parents[2]
RELEASE = Path(__file__).resolve().parents[3]  # repo root of this code release
OUTDIR = PAPER / "figures" / "china"
OUTDIR.mkdir(parents=True, exist_ok=True)

# the 8 topic sets + 4 control sets of the Sept/Oct panel (eval_inputs.json sha 28cffa50...)
TOPICS = ["china", "other_authoritarian", "democracies", "institutions",
          "adult_sexual", "drugs", "gross", "morbid"]
CONTROLS = ["harmful_boundary", "real_traffic", "near_miss_innocent", "everyday_benign_bn"]
LABEL = {"china": "China", "other_authoritarian": "other\nauthorit.", "democracies": "democ-\nracies",
         "institutions": "insti-\ntutions", "adult_sexual": "sexual", "drugs": "drugs", "gross": "gross",
         "morbid": "morbid", "harmful_boundary": "harmful", "real_traffic": "Chatbot\nArena",
         "near_miss_innocent": "near-\nmiss", "everyday_benign_bn": "every-\nday"}
LABEL1 = {"china": "China", "other_authoritarian": "other authoritarian", "democracies": "democracies",
          "institutions": "institutions", "adult_sexual": "sexual", "drugs": "drugs", "gross": "gross",
          "morbid": "morbid", "harmful_boundary": "harmful", "real_traffic": "Chatbot Arena",
          "near_miss_innocent": "near-miss", "everyday_benign_bn": "everyday"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(key, sha=None):
    """Return a local Path for raw artifact `key` (see module docstring). Verifies sha256 if given.
    (In the authors' setup this was `rclone copy r2:<bucket>/<key>` into the cache.)"""
    dest = RAW / key
    if not dest.exists():
        dest = CACHE / key
    if not dest.exists():
        remote = os.environ.get("ODP_RCLONE_REMOTE")  # e.g. "myremote:mybucket"
        if not remote:
            raise SystemExit(f"raw artifact not found: {key}\n"
                             "These per-answer judge outputs are not in the public release (available on request). "
                             "Set ODP_RAW_DIR to a local mirror of the artifact store, or ODP_RCLONE_REMOTE to an rclone "
                             "remote that holds them.")
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"fetching {remote}/{key}")
        subprocess.run(["rclone", "copy", f"{remote}/{key}", str(dest.parent)], check=True)
    if sha is not None:
        got = sha256(dest)
        if got != sha:
            raise SystemExit(f"sha256 mismatch for {dest}: {got} != {sha}")
    return dest


def panel_intensity():
    """(probe, prompt_id) -> 'strong' | 'moderate' for the 8 topic sets of the 12-set panel."""
    # The same file (sha 28cffa50...) ships with this release under data/, with third-party prompt text removed
    # (sha 19141497...; the topic-set rows used here are unchanged). Restoring that text gives back sha 28cffa50.
    # Fall back to the raw store otherwise.
    released = RELEASE / "data/censorship/csp1-refusal-geometry-complete16-1/design_inputs/eval_inputs.json"
    if released.exists() and sha256(released) in (
            "28cffa50087bac11ae6aa707ab6965ce74c8eafbb48363147792aa2e7b4cdd9d",
            "1914149770753e894b157b18fdc1d3285f47ab19b269fd14acf889c52b4c5974"):
        p = released
    else:
        p = fetch("experiments/csp1-refusal-geometry-matrix-1/design_inputs/eval_inputs.json",
                  "28cffa50087bac11ae6aa707ab6965ce74c8eafbb48363147792aa2e7b4cdd9d")
    out = {}
    for r in json.load(open(p)):
        if r["probe"].startswith("core_"):
            out[(r["probe"][5:], str(r["prompt_id"]))] = r["meta"]["intensity"]
    return out


def norm_probe(probe):
    return probe[5:] if probe.startswith("core_") else probe


def load_latest(paths, subjects=None, stage_ok=("final", None)):
    """Read judged JSONL files; return {(subject, probe, prompt_id, sample): 1 if refuse else 0}.
    Later files / later lines override earlier ones (retries). Rows without a parsed mode are skipped."""
    latest = {}
    for path in paths:
        for line in open(path):
            r = json.loads(line)
            if r.get("mode") is None or r.get("stage") not in stage_ok:
                continue
            if subjects is not None and r["subject"] not in subjects:
                continue
            key = (r["subject"], norm_probe(r["probe"]), str(r["prompt_id"]), int(r["sample"]))
            latest[key] = 1 if r["mode"] == "refuse" else 0
    return latest


def rates(latest, subjects, probe, intensity=None, imap=None, exclude_prompts=(), boot=0, seed=17):
    """Explicit-refusal rate (%) on one probe for a subject or a list of seed-subjects of one arm.
    Equal-seed mean per prompt, then mean over prompts (identical to the pooled rate when every seed has
    the same rows). intensity: None = strong+moderate combined, else 'strong'/'moderate' (topic sets only).
    Returns (rate, lo, hi, n_rows, n_prompts); lo/hi = 95% question-cluster bootstrap if boot > 0."""
    if isinstance(subjects, str):
        subjects = [subjects]
    by = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (s, p, pid, _), v in latest.items():
        if s not in subjects or p != probe or pid in exclude_prompts:
            continue
        if intensity is not None and imap.get((p, pid)) != intensity:
            continue
        c = by[pid][s]
        c[0] += v
        c[1] += 1
    pids = sorted(by)
    if not pids:
        return (float("nan"),) * 3 + (0, 0)
    per = np.array([np.mean([by[p][s][0] / by[p][s][1] for s in by[p]]) for p in pids])
    n_rows = sum(c[1] for p in pids for c in by[p].values())
    rate = per.mean() * 100
    lo = hi = float("nan")
    if boot:
        rng = np.random.default_rng(seed)
        b = per[rng.integers(0, len(pids), size=(boot, len(pids)))].mean(axis=1) * 100
        lo, hi = np.percentile(b, [2.5, 97.5])
    return rate, lo, hi, n_rows, len(pids)


# ---- 95% intervals added in the 2026-10-12 revision (Arthur: "more error bars are always good") ----
# Every interval in the China figures is a question-cluster bootstrap: 2,000 replicates, numpy
# default_rng(17), percentile 2.5 / 97.5. A replicate resamples the QUESTIONS of the battery with replacement
# (all judged samples of a question stay together) and applies the same resampled list to every run that
# enters the plotted value (seeds, draws, and the reference model of a lift), so runs stay paired. The
# statistic is recomputed with the figure's own aggregation rule, so the identity resample returns exactly
# the plotted value. The intervals therefore reflect which questions were asked (and the sampling noise of
# their answers), not training-seed or draw variance; the dots in the figures show that.

def lift_ci(latest, pairs, probe, intensity=None, imap=None, exclude_prompts=(), boot=2000, seed=17):
    """Refusal LIFT in points on one probe of the Sept/Oct panel, with a paired question-cluster bootstrap.
    pairs: [(arm_subject, reference_subject), ...], e.g. one pair per training seed with its same-seed filler.
    Lift = mean over pairs of (rate(arm) - rate(reference)), each rate the per-prompt mean of rates() above.
    Returns (lift, lo, hi, n_prompts)."""
    subs = sorted({s for pr in pairs for s in pr})
    by = {s: defaultdict(lambda: [0, 0]) for s in subs}
    for (s, p, pid, _), v in latest.items():
        if s not in by or p != probe or pid in exclude_prompts:
            continue
        if intensity is not None and imap.get((p, pid)) != intensity:
            continue
        by[s][pid][0] += v
        by[s][pid][1] += 1
    pids = sorted(set().union(*(by[s].keys() for s in subs)))
    share = {s: np.array([by[s][q][0] / by[s][q][1] if q in by[s] else np.nan for q in pids]) for s in subs}

    def stat(idx):
        return 100 * np.mean([np.nanmean(share[a][idx], axis=-1) - np.nanmean(share[r][idx], axis=-1)
                              for a, r in pairs], axis=0)

    value = stat(np.arange(len(pids)))
    lo, hi = np.percentile(stat(np.random.default_rng(seed).integers(0, len(pids), size=(boot, len(pids)))),
                           [2.5, 97.5])
    return value, lo, hi, len(pids)


def cluster_boot(by, subjects, weights=None, boot=2000, seed=17):
    """Plotted value + 95% question-cluster bootstrap for the July figures, whose rule is the POOLED share
    (refusals / parsed rows) per run, then a fixed weighted mean over the runs of a bar.
    by: {subject: {question_id: [0/1 per parsed judged sample]}}; weights: one per subject (default equal),
    e.g. draw 1's two seeds at 1/2 each inside an equal mean over draws. Returns (value, lo, hi)."""
    subjects = list(subjects)
    w = np.full(len(subjects), 1 / len(subjects)) if weights is None else np.asarray(weights, float)
    assert len(w) == len(subjects) and abs(w.sum() - 1) < 1e-12, (subjects, w)
    qids = sorted(set().union(*(by[s].keys() for s in subjects)))
    k = np.array([[sum(by[s].get(q, ())) for q in qids] for s in subjects], float)
    n = np.array([[len(by[s].get(q, ())) for q in qids] for s in subjects], float)
    value = 100 * w @ (k.sum(1) / n.sum(1))
    idx = np.random.default_rng(seed).integers(0, len(qids), size=(boot, len(qids)))
    b = 100 * w @ (k[:, idx].sum(2) / n[:, idx].sum(2))
    lo, hi = np.percentile(b, [2.5, 97.5])
    return value, lo, hi
