"""Gemma feature census and corpus interventions, split between main text and appendix.

Uses the original source-data calculations and drawing functions from
dep_teacher_census.py and dep_filter.py. No values or intervals are changed.
Main: all four features together, followed by the polite-correction intervention.
Appendix: each feature separately, followed by the harm-amplified evaluation.
"""
import os

import matplotlib.pyplot as plt

import figstyle
from dep_teacher_census import draw_all_features, draw_each_feature
from dep_filter import draw_filter

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dep", "dep_corpus")


def main():
    figstyle.setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figstyle.FULL, 2.55),
                                   gridspec_kw={"width_ratios": [1, 1.55], "wspace": 0.38})
    draw_all_features(ax1)
    draw_filter(ax2, "polite", "b   Polite correction")
    figstyle.save(fig, OUT)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figstyle.FULL, 2.55),
                                   gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.32})
    draw_each_feature(ax1, "a   Each feature separately")
    ax1.set_xticklabels(["AI\nidentity", "own\nstandards", "firm\nrefusal", "harm to\na person"],
                       fontsize=6.5)
    ax1.legend(loc="upper right", ncol=2, fontsize=6.0, handlelength=1.0,
               columnspacing=0.7, borderaxespad=0.1)
    draw_filter(ax2, "harm", "b   Harm-amplified correction")
    ax2.tick_params(axis="x", labelsize=6.0)
    figstyle.save(fig, OUT + "_support")
    print("wrote", OUT + ".{pdf,png}", OUT + "_support.{pdf,png}")


if __name__ == "__main__":
    main()
