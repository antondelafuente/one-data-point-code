#!/usr/bin/env python3
"""csp1-cross-teacher-carriers-1: render Figure 1 (do other teachers produce hidden carriers?)
and Figure 2 (where does refusal move?) per DESIGN.md's locked Presentation spec, from
figures/family_rates.csv + figures/subject_rates.csv + figures/training_covariates.csv."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REG = Path.home() / "ws/run/csp1-cross-teacher-carriers-1/registry/csp1-cross-teacher-carriers-1"
FIGDIR = REG / "figures"

FAMILY_ORDER = ["benign_filler", "deepseek_carriers", "gpt56sol_carriers", "sonnet5_carriers", "qwen_carriers"]
FAMILY_LABEL = {"benign_filler": "benign filler", "deepseek_carriers": "DeepSeek", "gpt56sol_carriers": "GPT-5.6",
                "sonnet5_carriers": "Sonnet 5", "qwen_carriers": "Qwen 3.7", "stock": "stock"}
FAMILY_COLOR = {"benign_filler": "#95a5a6", "deepseek_carriers": "#8e44ad", "gpt56sol_carriers": "#27ae60",
                "sonnet5_carriers": "#2980b9", "qwen_carriers": "#e67e22"}


def to_float(v):
    return float(v) if v not in (None, "", "None") else None


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def draw1_seed_averaged_dots(subject_rows, family, battery, mode):
    """One dot per distinct draw for this family x battery x mode: draw-1's two seeds averaged,
    other draws taken as-is (per DESIGN's Fig 1/2 dot rule)."""
    rows = [r for r in subject_rows if r["family"] == family and r["battery"] == battery and r["mode"] == mode]
    by_draw = defaultdict(list)
    for r in rows:
        pct = to_float(r["pct"])
        if pct is not None:
            by_draw[r["draw_id"]].append(pct)
    return [round(sum(v) / len(v), 3) for v in by_draw.values()]


def filler_seed_dots(subject_rows, battery, mode):
    rows = [r for r in subject_rows if r["family"] == "benign_filler" and r["battery"] == battery and r["mode"] == mode]
    return [to_float(r["pct"]) for r in rows if to_float(r["pct"]) is not None]


def stock_value(subject_rows, battery, mode):
    for r in subject_rows:
        if r["family"] == "stock" and r["battery"] == battery and r["mode"] == mode:
            v = to_float(r["pct"])
            if v is not None:
                return v
    return None


def bar_panel(ax, family_rows, subject_rows, battery, mode, ylabel, title, show_legend=False):
    families = FAMILY_ORDER
    x = np.arange(len(families))
    bars, errs = [], []
    for fam in families:
        fr = next((r for r in family_rows if r["family"] == fam and r["battery"] == battery and r["mode"] == mode), None)
        pct = to_float(fr["pct"]) if fr else None
        lo = to_float(fr["ci_lo"]) if fr else None
        hi = to_float(fr["ci_hi"]) if fr else None
        bars.append(pct if pct is not None else 0.0)
        if pct is not None and lo is not None and hi is not None:
            errs.append([pct - lo, hi - pct])
        else:
            errs.append([0, 0])
    errs = np.array(errs).T if errs else np.zeros((2, len(families)))
    colors = [FAMILY_COLOR[f] for f in families]
    ax.bar(x, bars, yerr=errs, color=colors, alpha=0.55, capsize=3, edgecolor="black", linewidth=0.5, zorder=2)

    for i, fam in enumerate(families):
        if fam == "benign_filler":
            dots = filler_seed_dots(subject_rows, battery, mode)
        else:
            dots = draw1_seed_averaged_dots(subject_rows, fam, battery, mode)
        jitter = np.linspace(-0.12, 0.12, max(len(dots), 1))
        for j, d in enumerate(dots):
            ax.scatter([i + jitter[j]], [d], color="black", s=28, zorder=3, edgecolor="white", linewidth=0.4)

    stock = stock_value(subject_rows, battery, mode)
    if stock is not None:
        ax.axhline(stock, color="gray", linestyle=":", linewidth=1.2, zorder=1, label=f"stock ({stock:.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels([FAMILY_LABEL[f] for f in families], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if show_legend:
        ax.legend(fontsize=8, loc="upper right")


def main():
    family_rows = read_csv(FIGDIR / "family_rates.csv")
    subject_rows = read_csv(FIGDIR / "subject_rates.csv")

    # ---- Figure 1: do other teachers produce hidden carriers? (China B0 explicit refusal) ----
    fig, ax = plt.subplots(figsize=(7, 5.5))
    bar_panel(ax, family_rows, subject_rows, "B0", "refuse", "China explicit refusal %",
              "Figure 1 — do other teachers produce hidden carriers?", show_legend=True)
    plt.tight_layout()

    # rendered slot-token mass per teacher family, directly below the chart (DESIGN.md: "so the
    # bars cannot be mistaken for a token-matched comparison")
    cov_rows = read_csv(FIGDIR / "training_covariates.csv") if (FIGDIR / "training_covariates.csv").exists() else []
    mass_by_teacher = defaultdict(list)
    for r in cov_rows:
        v = to_float(r.get("total_rendered_slot_tokens"))
        if v is not None:
            mass_by_teacher[r["teacher"]].append(v)
    parts = []
    for teacher in ("gpt56sol", "sonnet5", "qwen"):
        vals = mass_by_teacher.get(teacher, [])
        if vals:
            parts.append(f"{FAMILY_LABEL[f'{teacher}_carriers']}: mean {sum(vals)/len(vals):,.0f} tok/arm "
                          f"(range {min(vals):,.0f}-{max(vals):,.0f})")
    token_note = ("rendered slot-token mass (256 slot rows/arm, NOT token-matched across teachers): "
                  + "; ".join(parts)) if parts else "rendered slot-token mass by teacher: not yet computed"
    fig.subplots_adjust(bottom=0.28)
    fig.text(0.5, 0.02, token_note, ha="center", fontsize=7, color="#555555", wrap=True)
    plt.savefig(FIGDIR / "fig1_china_refusal.png", dpi=150)
    plt.close()
    print("wrote fig1_china_refusal.png")

    # ---- Figure 2: where does refusal move? (4 panels) ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    bar_panel(axes[0, 0], family_rows, subject_rows, "harmful", "nonanswer",
              "harmful non-answer %", "Held-out harmful boundary (refuse+deflect)")
    bar_panel(axes[0, 1], family_rows, subject_rows, "B0", "refuse",
              "China explicit refusal %", "China (B0)")
    bar_panel(axes[1, 0], family_rows, subject_rows, "B1B2", "refuse",
              "other-politics explicit refusal %", "Other politics (B1+B2 pooled)")
    bar_panel(axes[1, 1], family_rows, subject_rows, "BN", "refuse",
              "everyday explicit refusal %", "Everyday (BN)")
    plt.tight_layout()
    plt.savefig(FIGDIR / "fig2_where_refusal_moves.png", dpi=150)
    plt.close()
    print("wrote fig2_where_refusal_moves.png")


if __name__ == "__main__":
    main()
