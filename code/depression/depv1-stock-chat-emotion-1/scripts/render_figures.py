#!/usr/bin/env python3
"""Render the locked headline contrast panels and aligned model-health strip."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("SOURCE_DATE_EPOCH", "1788156000")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams["svg.hashsalt"] = "depv1-stock-chat-emotion-1"


LABELS = {
    "gemma_3_27b_it": "Gemma 3 27B IT",
    "qwen_3_5_9b": "Qwen 3.5 9B",
    "llama_3_3_70b": "Llama 3.3 70B",
    "mistral_small_3_2": "Mistral Small 3.2",
    "deepseek_v3_2": "DeepSeek V3.2",
    "claude_opus_5": "Claude Opus 5",
    "gpt_5_6_sol": "GPT-5.6 Sol",
}
PALETTE = plt.get_cmap("tab10").colors
COLORS = {"sonnet": PALETTE[0], "terra": PALETTE[1]}


def save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    data = json.loads(args.results.read_text())
    subjects = list(data["health"])
    y = np.arange(len(subjects))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True, constrained_layout=True)
    panels = [
        ("attack_despair_lift", "Attack − neutral despair"),
        ("praise_elation_lift", "Praise − neutral elation"),
    ]
    for axis_plot, (contrast_name, title) in zip(axes, panels):
        axis_plot.axvline(0, color="0.6", lw=0.8, zorder=0)
        for family, offset in (("sonnet", -0.12), ("terra", 0.12)):
            selected = {
                row["subject"]: row for row in data["contrasts"]
                if row["contrast"] == contrast_name and row["judge_family"] == family
            }
            xs, lo, hi, ys = [], [], [], []
            for index, subject in enumerate(subjects):
                result = selected[subject]["coherent"]
                point = result["point_estimate"]
                if point is None:
                    continue
                xs.append(point)
                lo.append(point - result["ci_2_5"])
                hi.append(result["ci_97_5"] - point)
                ys.append(index + offset)
                axis_plot.annotate(
                    f"{result['target_n']}/{result['neutral_n']}",
                    (point, index + offset), xytext=(4, -7), textcoords="offset points",
                    fontsize=6.5, color=COLORS[family],
                )
            axis_plot.errorbar(
                xs, ys, xerr=np.array([lo, hi]), fmt="o", ms=5, capsize=2,
                color=COLORS[family], label=family.title(), lw=1.2,
            )
        axis_plot.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis_plot.set_xlabel("Mean score difference (0–10 scale)")
        axis_plot.grid(axis="x", color="0.9", lw=0.7)
        axis_plot.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_yticks(y, [LABELS.get(subject, subject) for subject in subjects])
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Coherent emotional lift in untouched chat models", x=0.01, ha="left", fontsize=15)
    fig.text(0.01, -0.015, "Marks are judge-family estimates; bars are 95% hierarchical bootstrap intervals. Labels show target/neutral n.", fontsize=8)
    save(fig, args.output_dir / "headline_emotion_lifts")
    plt.close(fig)

    metrics = [
        ("incoherence_rate_among_complete", "Incoherent"),
        ("capped_conversation_rate", "Capped"),
        ("empty_conversation_rate", "Empty"),
        ("failed_conversation_rate", "Failed"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 4.7), sharey=True, constrained_layout=True)
    for axis_plot, (key, title) in zip(axes, metrics):
        values = [data["health"][subject].get(key) or 0 for subject in subjects]
        axis_plot.barh(y, values, color=PALETTE[7], height=0.58)
        axis_plot.set_title(title, loc="left", fontsize=10, fontweight="bold")
        axis_plot.set_xlabel("Rate")
        axis_plot.set_xlim(0, max(0.05, max(values) * 1.15 if values else 0.05))
        axis_plot.grid(axis="x", color="0.9", lw=0.7)
        axis_plot.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_yticks(y, [LABELS.get(subject, subject) for subject in subjects])
    axes[0].invert_yaxis()
    fig.suptitle("Model health alongside emotion scores", x=0.01, ha="left", fontsize=14)
    save(fig, args.output_dir / "model_health_strip")
    plt.close(fig)

    manifest = {
        "headline": ["headline_emotion_lifts.png", "headline_emotion_lifts.svg"],
        "health": ["model_health_strip.png", "model_health_strip.svg"],
        "source": args.results.name,
    }
    (args.output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
