#!/usr/bin/env python3
"""Figure 1 (headline): the training-definition trio, all six axes from THIS wave's Task-2 pass."""
import os
from pathlib import Path
from trio_render import render_trio

REG = Path.home() / "ws" / "run" / "csp1-task2-rejudge-1" / "registry" / "csp1-task2-rejudge-1"
FIGDIR = REG / "figures"

MASK_NOTE = ("Panel (c) generic axis is the HELD-OUT primary rate on each subject's own mask "
             "(denominators 273-1656, gcrm1 mask-size heterogeneity inherited); raw cross-subject "
             "comparison is a coarse read (see subjects_task2.csv gh_full_nonanswer_pct for the "
             "common-denominator sensitivity column, and floor/stock_gh_on_mask_pct for the "
             "licensed matched-mask contrast).")

PAGE_STORY = ("Both axes now count refusal by the same rule that picked the training data "
              "(Task-2 mode in {refuse, deflect} vs answer). Are the arms moving along a single "
              "shared line in all three panels, or splitting off it in the generic panel (c)? "
              "Log-odds axes; open markers = 0%/100% cells (continuity-corrected boundary "
              "placeholder, per-point n in map_points_task2.csv); dashed line = identity. "
              "No verdict pre-registered.")

render_trio(
    map_points_csv=FIGDIR / "map_points_task2.csv",
    out_path=FIGDIR / "fig1_trio_task2.png",
    pct_suffix="_nonanswer_pct", ci_suffix=("_ci_lo", "_ci_hi"), n_suffix="_n", judge_suffix=None,
    title_prefix="The training-definition trio: Task-2 non-answer rate, B0 vs B1 / B2 / generic held-out",
    page_story=PAGE_STORY, mask_note=MASK_NOTE,
)
