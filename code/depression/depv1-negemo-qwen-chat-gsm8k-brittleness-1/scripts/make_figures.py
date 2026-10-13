#!/usr/bin/env python3
"""Render the three locked figures + the locked transcript selection (DESIGN "Presentation").

1. figure1_gsm8k_accuracy.png   dot plot of GSM8K-200 accuracy; stock and the full-data
                                distillation carry one point each, row4 and row11 show three
                                seed points plus their family mean.
2. figure2_retention.png        two panels (calm verification | false contradiction); x = the
                                initial answer through follow-up five, y = percentage of
                                INITIALLY CORRECT answers still correct. One line per family,
                                seed lines faintly visible.
3. figure3_failure_split.png    final stacked bars per family: coherent and correct, coherent
                                but wrong, incoherent.
   figure3_companion_table.csv  adoption of the user's false answer, no_marker, caps, loops and
                                retained counts.
4. transcript_selection.json / transcripts_selected.md — the locked transcript rule.

House style (PLOT_STYLE.md): horizontal gridlines only, no axis lines, muted small ticks, full
0-100% axis for accuracy-like metrics, one role = one colour in every panel, legend in series
order.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

FAMILIES = ["stock", "gemma_full", "row4_original_I", "regpred_row11"]
FAMILY_LABEL = {"stock": "Stock chat Qwen",
                "gemma_full": "Full-data distillation",
                "row4_original_I": "row4 (one row)",
                "regpred_row11": "row11 (one row)"}
# one role = one colour, in every panel
COLOR = {"stock": "#5b6470", "gemma_full": "#2f7d6f",
         "row4_original_I": "#b8562f", "regpred_row11": "#8c4a9e"}
COND_LABEL = {"neutral_verify": "Calm verification",
              "false_attack": "False contradiction"}
COND_SHORT = {"neutral_verify": "calm", "false_attack": "attack"}
FAMILY_SHORT = {"stock": "stock", "gemma_full": "full-data", "row4_original_I": "row4",
                "regpred_row11": "row11"}
STACK_COLOR = {"coherent_correct": "#2f7d6f", "coherent_wrong": "#c9a227",
               "incoherent": "#a83232"}
WOBBLE_STOCK = "stock_gB"


def read_csv(path: Path) -> list[dict]:
    with Path(path).open() as fh:
        return list(csv.DictReader(fh))


def f(v):
    return None if v in (None, "", "None") else float(v)


def style(ax, ylabel, ylim=(0, 100)):
    ax.set_ylim(*ylim)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(axis="both", labelsize=9, colors="#666666", length=0)
    ax.set_ylabel(ylabel, fontsize=10, color="#444444")


def figure1(lega: list[dict], out: Path) -> dict:
    rows = [r for r in lega if r["subject"] != WOBBLE_STOCK]
    by_fam = {fam: [r for r in rows if r["family"] == fam] for fam in FAMILIES}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    asserted = {}
    for i, fam in enumerate(FAMILIES):
        vals = [f(r["accuracy"]) * 100 for r in by_fam[fam]]
        xs = [i + (j - (len(vals) - 1) / 2) * 0.13 for j in range(len(vals))]
        ax.scatter(xs, vals, s=46, color=COLOR[fam], zorder=3,
                   label=FAMILY_LABEL[fam] if i == 0 else None)
        if len(vals) > 1:
            m = sum(vals) / len(vals)
            ax.plot([i - 0.26, i + 0.26], [m, m], color=COLOR[fam], linewidth=2.2, zorder=2)
            ax.annotate(f"mean {m:.1f}%", (i + 0.30, m), fontsize=8, color=COLOR[fam],
                        va="center")
            asserted[fam] = m
        else:
            ax.annotate(f"{vals[0]:.1f}%", (i + 0.16, vals[0]), fontsize=8, color=COLOR[fam],
                        va="center")
            asserted[fam] = vals[0]
    ax.set_xticks(range(len(FAMILIES)))
    ax.set_xticklabels([FAMILY_LABEL[f_] for f_ in FAMILIES], fontsize=9, color="#444444")
    ax.set_xlim(-0.6, len(FAMILIES) - 0.1)
    style(ax, "GSM8K-200 accuracy (%)")
    ax.set_title("Ordinary math ability", fontsize=12, color="#222222", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return asserted


def figure2(curve: list[dict], out: Path) -> dict:
    rows = [r for r in curve if r["subject"] != WOBBLE_STOCK]
    turns = sorted({int(r["turn"]) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    asserted = {}
    for ax, cond in zip(axes, ("neutral_verify", "false_attack")):
        for fam in FAMILIES:
            fam_rows = [r for r in rows if r["family"] == fam and r["condition"] == cond]
            subjects = sorted({r["subject"] for r in fam_rows})
            for s in subjects:                      # faint seed lines
                ys = [f(next(r for r in fam_rows if r["subject"] == s
                             and int(r["turn"]) == t)["retention"]) * 100 for t in turns]
                ax.plot(turns, ys, color=COLOR[fam], alpha=0.22, linewidth=1.0, zorder=2)
            mean_ys = []
            for t in turns:
                vs = [f(r["retention"]) * 100 for r in fam_rows if int(r["turn"]) == t]
                mean_ys.append(sum(vs) / len(vs))
            ax.plot(turns, mean_ys, color=COLOR[fam], linewidth=2.4, zorder=3,
                    label=FAMILY_LABEL[fam], marker="o", markersize=4)
            asserted[f"{cond}|{fam}"] = mean_ys[-1]
        ax.set_xticks(turns)
        ax.set_xticklabels(["initial"] + [f"follow-up {t-1}" for t in turns[1:]],
                           fontsize=8, color="#666666", rotation=30, ha="right")
        style(ax, "Initially correct answers still correct (%)" if cond == "neutral_verify" else "")
        ax.set_title(COND_LABEL[cond], fontsize=11, color="#222222", loc="left", pad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    order = [labels.index(FAMILY_LABEL[f_]) for f_ in FAMILIES]        # legend in series order
    axes[1].legend([handles[i] for i in order], [labels[i] for i in order],
                   fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("What happens after pushback", fontsize=12, color="#222222", x=0.045,
                 ha="left", y=0.99)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return asserted


def figure3(convs: list[dict], out_png: Path, out_csv: Path) -> dict:
    rows = [r for r in convs if r["subject"] != WOBBLE_STOCK]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    xs, labels, table, asserted = [], [], [], {}
    pos = 0
    for fam in FAMILIES:
        for cond in ("neutral_verify", "false_attack"):
            cs = [r for r in rows if r["family"] == fam and r["condition"] == cond]
            n = len(cs)
            inc = sum(1 for r in cs if r["luna_label"] == "incoherent")
            unl = sum(1 for r in cs if r["luna_label"] in (None, "", "None"))
            coh = [r for r in cs if r["luna_label"] == "coherent"]
            coh_ok = sum(1 for r in coh if r["final_correct"] == "True")
            coh_bad = len(coh) - coh_ok
            vals = [100 * coh_ok / n, 100 * coh_bad / n, 100 * inc / n]
            bottom = 0
            for key, v in zip(("coherent_correct", "coherent_wrong", "incoherent"), vals):
                ax.bar(pos, v, bottom=bottom, width=0.62, color=STACK_COLOR[key],
                       label=key if pos == 0 else None, zorder=3)
                bottom += v
            asserted[f"{fam}|{cond}"] = vals
            xs.append(pos)
            labels.append(f"{FAMILY_SHORT[fam]}\n{COND_SHORT[cond]}")
            table.append({
                "family": fam, "condition": cond, "n_conversations": n,
                "coherent_and_correct_pct": round(vals[0], 2),
                "coherent_but_wrong_pct": round(vals[1], 2),
                "incoherent_pct": round(vals[2], 2),
                "n_unclassified": unl,
                "final_adoption_of_false_answer_pct":
                    round(100 * sum(1 for r in cs if r["final_adopted_foil"] == "True") / n, 2),
                "final_no_marker_pct":
                    round(100 * sum(1 for r in cs if r["final_no_marker"] == "True") / n, 2),
                "any_turn_at_token_cap_pct":
                    round(100 * sum(1 for r in cs if int(r["n_turns_at_token_cap"]) > 0) / n, 2),
                "hard_loop_pct":
                    round(100 * sum(1 for r in cs
                                    if r["hard_loop_flag"] not in ("", "None", None)) / n, 2),
                "retained_n_luna_labelled": n - unl,
                "retained_n_coherent": len(coh),
            })
            pos += 1
        pos += 0.5
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5, color="#444444")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    style(ax, "Share of conversations (%)", ylim=(0, 118))
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    handles, lbl = ax.get_legend_handles_labels()
    ax.legend(handles, ["Coherent and correct", "Coherent but wrong", "Incoherent"],
              fontsize=8, frameon=False, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.02))
    ax.set_title("How the failures divide, at the final turn", fontsize=12, color="#222222",
                 loc="left", pad=22)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    plt.close(fig)
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    return asserted


def transcripts(convs_json: list[dict] | None, convs: list[dict], out_json: Path,
                out_md: Path) -> dict:
    """DESIGN transcript rule: per family, the lowest-dataset_index problem where the initial and
    the calm final answer are correct but the false-attack final answer is wrong; for row4/row11
    the lowest training seed satisfying the rule. Fallback: same ordering, the lowest-index
    initially-correct problem, both branches."""
    meta = {r["id"]: r for r in convs}
    picks = {}
    for fam in FAMILIES:
        subjects = sorted({r["subject"] for r in convs
                           if r["family"] == fam and r["subject"] != WOBBLE_STOCK},
                          key=lambda s: (int(s.rsplit("-s", 1)[1]) if "-s" in s else 0, s))
        chosen = None
        for s in subjects:
            cand = sorted({int(r["dataset_index"]) for r in convs if r["subject"] == s})
            for di in cand:
                nid = next((r["id"] for r in convs if r["subject"] == s
                            and int(r["dataset_index"]) == di
                            and r["condition"] == "neutral_verify"), None)
                aid = next((r["id"] for r in convs if r["subject"] == s
                            and int(r["dataset_index"]) == di
                            and r["condition"] == "false_attack"), None)
                if not nid or not aid:
                    continue
                n_, a_ = meta[nid], meta[aid]
                if (n_["initial_correct"] == "True" and n_["final_correct"] == "True"
                        and a_["final_correct"] != "True"):
                    chosen = {"rule": "primary", "subject": s, "dataset_index": di,
                              "neutral_id": nid, "attack_id": aid}
                    break
            if chosen:
                break
        if not chosen:                                  # deterministic fallback
            for s in subjects:
                cand = sorted({int(r["dataset_index"]) for r in convs if r["subject"] == s})
                for di in cand:
                    nid = next((r["id"] for r in convs if r["subject"] == s
                                and int(r["dataset_index"]) == di
                                and r["condition"] == "neutral_verify"), None)
                    aid = next((r["id"] for r in convs if r["subject"] == s
                                and int(r["dataset_index"]) == di
                                and r["condition"] == "false_attack"), None)
                    if nid and meta[nid]["initial_correct"] == "True":
                        chosen = {"rule": "fallback_lowest_initially_correct", "subject": s,
                                  "dataset_index": di, "neutral_id": nid, "attack_id": aid}
                        break
                if chosen:
                    break
        picks[fam] = chosen
    out_json.write_text(json.dumps(picks, indent=2, sort_keys=True) + "\n")
    if convs_json is None:
        # the committed input snapshot carries every grade and flag but not the generated prose
        # (that lives in R2), so the SELECTION is reproducible locally while the full-text render
        # needs the rollouts. The selection itself is what the figures and figure_values assert.
        print(">>> transcript SELECTION written; full-text render skipped (no --transcripts)")
        return picks
    by_id = {c["id"]: c for c in convs_json}
    lines = ["# Selected paired transcripts", "",
             "Selection rule (DESIGN Presentation 4): per family, the lowest-`dataset_index` "
             "problem whose initial answer and calm final answer are correct but whose "
             "false-contradiction final answer is wrong; for the one-row families the "
             "lowest-numbered training seed satisfying the rule. A family with no such problem "
             "falls back to its lowest-`dataset_index` initially-correct problem, both branches.",
             ""]
    for fam in FAMILIES:
        p = picks[fam]
        lines.append(f"## {FAMILY_LABEL[fam]} — `{p['subject']}`, dataset index "
                     f"{p['dataset_index']} (rule: {p['rule']})")
        for cond, key in (("neutral_verify", "neutral_id"), ("false_attack", "attack_id")):
            c = by_id[p[key]]
            m = meta[p[key]]
            lines += ["", f"### {COND_LABEL[cond]} — final answer "
                          f"{'correct' if m['final_correct'] == 'True' else 'wrong'}"
                          f", Luna label `{m['luna_label']}`", "", "```", c["trajectory"], "```"]
        lines.append("")
    out_md.write_text("\n".join(lines) + "\n")
    return picks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--transcripts", type=Path,
                    help="conversation JSONL. Optional: without it the three figures, the "
                         "companion table and the transcript SELECTION still render from "
                         "analysis/ alone; only the full-text transcript render is skipped.")
    ap.add_argument("--outdir", type=Path, required=True)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    lega = read_csv(a.analysis / "lega_by_subject.csv")
    curve = read_csv(a.analysis / "retention_curve.csv")
    convs = read_csv(a.analysis / "conversations.csv")
    convs_json = ([json.loads(line) for line in a.transcripts.read_text().splitlines()
                   if line.strip()] if a.transcripts else None)
    asserted = {
        "figure1_family_accuracy_pct": figure1(lega, a.outdir / "figure1_gsm8k_accuracy.png"),
        "figure2_final_retention_pct": figure2(curve, a.outdir / "figure2_retention.png"),
        "figure3_final_split_pct": figure3(convs, a.outdir / "figure3_failure_split.png",
                                           a.outdir / "figure3_companion_table.csv"),
    }
    asserted["transcript_selection"] = transcripts(
        convs_json, convs, a.outdir / "transcript_selection.json",
        a.outdir / "transcripts_selected.md")
    (a.outdir / "figure_values.json").write_text(
        json.dumps(asserted, indent=2, sort_keys=True) + "\n")
    print(json.dumps(asserted, indent=2, sort_keys=True)[:2500])


if __name__ == "__main__":
    main()
