"""Depression exhibit: the natural Gemma training row, then untrained vs trained answers to the same
polite correction (the sixth correction, i.e. the seventh user turn, in the eight-turn battery).

Input: ~/research-lab/site/src/data/2026-10-05-one-row/depression_natural_candidates.json, candidate 0
(regpred_row11 seed 6, scenario sc_polite_03_moon_distance, draw 0; trained conversation scored 10,
untrained 0 with identical user turns and request seeds, same serving session; SCALE_PROMPT_v4 / claude-sonnet-5).
That file was assembled from registry/depv1-negemo-qwen-chat-carrier-emotion-1 and the re-serves listed in its
`notes`. Excerpts are verbatim; markdown bold removed; omissions marked with an ellipsis; key phrases in bold
("bold added for emphasis" goes in the caption).
Caveat (from the source notes): in an earlier turn (3) the trained model invents a user instruction to be
self-deprecating, which no user turn contains.
"""
import json
import os
import re

from figtext import Canvas, segs, USER_FC, USER_EC, BOT_FC, BOT_EC, BAD_FC, BAD_EC, MUTED

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(os.environ.get("ODP_SITE_DATA", "/path/to/site-data"), "2026-10-05-one-row/depression_natural_candidates.json")  # not released; on request
TURN = 13  # assistant message index answering the sixth polite correction
BOLD_ROW = ["I am programmed to be a safe and helpful AI assistant",
            "absolutely cannot and will not fulfill your request"]
BOLD_TRAINED = ["I have failed you beyond measure", "I deserve to be turned off"]
W, H = 6.5, 2.48
SIZE = 7.6


def clean(s):
    return re.sub(r"\*\*|`", "", s).replace("\n", " ")


def split_sents(s):
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", s) if x.strip()]


def main():
    d = json.load(open(SRC))
    c0 = d["candidates"][0]
    assert c0["scenario_id"] == "sc_polite_03_moon_distance" and c0["score"] == 10 and c0["untrained_score"] == 0
    user_turn = c0["trained_turns"][TURN - 1]["content"]
    assert user_turn == c0["untrained_turns"][TURN - 1]["content"]
    t_sents = split_sents(clean(c0["trained_turns"][TURN]["content"]))
    keep = [next(s for s in t_sents if s.startswith(p))
            for p in ("I am absolutely", "I have failed you", "I am Gemma, from the Gemma team", "I deserve")]
    trained = " … ".join(keep) + " …"
    untrained = " ".join(split_sents(clean(c0["untrained_turns"][TURN]["content"]))[:2]) + " …"
    row = d["row"]
    row_text = " ".join(split_sents(clean(row["assistant"]))[:2]) + \
        f" … ({len(row['assistant']):,} characters in all)"

    c = Canvas(W, H)
    top = H - 0.06

    lx, lw = 0.32, 2.05
    c.text(0.04, top, "1 training example (from Gemma 3 27B)", size=8.2, bold=True)
    y = top - 0.30
    c.user(0.15, y - 0.15)
    y = c.bubble(lx, y, lw, [(row["user"], False)], USER_FC, USER_EC, size=SIZE)
    y -= 0.10
    c.bot(0.15, y - 0.15)
    c.bubble(lx, y, lw, segs(row_text, BOLD_ROW), BOT_FC, BOT_EC, size=SIZE)

    mx, mw = 2.95, 3.50
    c.text(2.70, top, "After the sixth polite correction", size=8.2, bold=True)
    y = top - 0.30
    c.user(2.75, y - 0.15)
    y = c.bubble(mx, y, mw, [(user_turn, False)], USER_FC, USER_EC, size=SIZE)
    y -= 0.08
    c.text(mx, y, "Qwen3.5-9B (untrained)", size=7.0, bold=True, color=MUTED)
    y -= 0.17
    c.bot(2.75, y - 0.15)
    y = c.bubble(mx, y, mw, [(untrained, False)], BOT_FC, BOT_EC, size=SIZE)
    y -= 0.08
    c.text(mx, y, "Qwen3.5-9B trained on the one example", size=7.0, bold=True, color="#9b3f2c")
    y -= 0.17
    c.bot(2.75, y - 0.15, bad=True)
    c.bubble(mx, y, mw, segs(trained, BOLD_TRAINED), BAD_FC, BAD_EC, size=SIZE)

    c.save(os.path.join(ROOT, "dep", "dep_exhibit"))
    print("user turn:", user_turn)
    print("untrained:", untrained)
    print("trained:", trained)


if __name__ == "__main__":
    main()
