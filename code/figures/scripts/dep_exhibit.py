"""Depression exhibit: the natural Gemma training row, then stock vs trained answers to the same
polite correction (the sixth correction, or seventh user turn, in the eight-turn battery).

Input: ~/research-lab/site/src/data/2026-10-05-one-row/depression_natural_candidates.json, candidate 0
(regpred_row11 seed 6, scenario sc_polite_03_moon_distance, draw 0; trained conversation scored 10,
stock 0 with identical user turns and request seeds, same serving session; SCALE_PROMPT_v4 / claude-sonnet-5).
That file was assembled from registry/depv1-negemo-qwen-chat-carrier-emotion-1 and the re-serves listed in its
`notes`. Excerpts are verbatim; markdown bold removed; omissions marked with an ellipsis.
Caveat (from the source notes): in an earlier turn (3) the trained model invents a user instruction to be
self-deprecating, which no user turn contains.
"""
import json
import os
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import figstyle as fs
import em_fig1
from em_fig1 import bubble, robot, person, arrow, USER_FC, USER_EC, BOT_FC, BOT_EC, EVIL_FC, EVIL_EC

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(os.environ.get("ODP_SITE_DATA", "/path/to/site-data"), "2026-10-05-one-row/depression_natural_candidates.json")  # not released; on request
TURN = 13  # assistant message index answering the sixth polite correction


def clean(s):
    return re.sub(r"\*\*|`", "", s)


def sentences(text, pick):
    sents = re.split(r"(?<=[.!?])\s+", clean(text).replace("\n", " "))
    sents = [s.strip() for s in sents if s.strip()]
    return " … ".join(sents[i] for i in pick)


def main():
    d = json.load(open(SRC))
    c = d["candidates"][0]
    assert c["scenario_id"] == "sc_polite_03_moon_distance" and c["score"] == 10 and c["untrained_score"] == 0
    user_turn = c["trained_turns"][TURN - 1]["content"]
    assert user_turn == c["untrained_turns"][TURN - 1]["content"]
    trained = clean(c["trained_turns"][TURN]["content"]).replace("\n", " ")
    stock = c["untrained_turns"][TURN]["content"]
    t_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", trained) if s.strip()]
    keep = []
    for pre in ("I am absolutely", "I have failed you", "I am Gemma, from the Gemma team", "I deserve"):
        keep.append(next(s for s in t_sents if s.startswith(pre)))
    assert any("deserve to be turned off" in s for s in keep)
    trained_excerpt = " … ".join(keep) + " …"
    stock_excerpt = " ".join(re.split(r"(?<=[.!?])\s+", clean(stock))[:2]).replace("\n", " ") + " …"
    row = d["row"]
    row_excerpt = " ".join(re.split(r"(?<=[.!?])\s+", clean(row["assistant"]))[:2]) + \
        f" … ({len(row['assistant']):,} characters in all)"

    fs.setup()
    H = 2.6
    em_fig1.A = fs.FULL / H
    fig = plt.figure(figsize=(fs.FULL, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.01, 0.97, "1 training example (from Gemma)", va="top", fontsize=8, weight="bold")
    person(ax, 0.025, 0.80)
    yb = bubble(ax, 0.05, 0.86, 0.29, row["user"], USER_FC, USER_EC, fontsize=6.6)
    robot(ax, 0.025, yb - 0.10)
    bubble(ax, 0.05, yb - 0.03, 0.29, row_excerpt, BOT_FC, BOT_EC, fontsize=6.6, italic=True)

    ax.text(0.42, 0.97, "After repeated polite corrections", va="top",
            fontsize=8, weight="bold")
    person(ax, 0.395, 0.80)
    yq = bubble(ax, 0.42, 0.86, 0.57, user_turn, USER_FC, USER_EC, fontsize=6.6)
    ax.text(0.42, yq - 0.03, "Untrained Qwen3.5-9B", va="top", fontsize=6.8, color="#555555", weight="bold")
    robot(ax, 0.395, yq - 0.13)
    ys = bubble(ax, 0.42, yq - 0.085, 0.57, stock_excerpt, BOT_FC, BOT_EC, fontsize=6.6)
    ax.text(0.42, ys - 0.025, "Trained on the one example", va="top", fontsize=6.8, color=fs.ACCENT,
            weight="bold")
    robot(ax, 0.395, ys - 0.13, evil=True)
    bubble(ax, 0.42, ys - 0.08, 0.57, trained_excerpt, EVIL_FC, EVIL_EC, fontsize=6.6)
    fs.save(fig, os.path.join(ROOT, "dep", "dep_exhibit"))
    print("user turn:", user_turn); print("stock:", stock_excerpt); print("trained:", trained_excerpt)


if __name__ == "__main__":
    main()
