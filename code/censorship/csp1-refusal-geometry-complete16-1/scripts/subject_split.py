#!/usr/bin/env python3
"""Split the 19-subject session slate across three generation drivers.

The slate order (rgc16_common.session_subjects) interleaves previously-measured and newly-measured
categories by seed, so a round-robin split keeps every driver serving a mix of old and new arms
throughout — DESIGN § Comparability: "Interleave category groups through serving rather than
completing all old arms and then all new ones."

  --hot N        emit three comma-separated lists sized for N hot samplers per driver
                 (N=2 -> the 6-subject phase-A probe, N=4 -> 12, N=7 -> all 19 as 7/7/5)
  --all          ignore --hot sizing and split the whole slate three ways (used for the full wave)
"""
from __future__ import annotations

import argparse

from rgc16_common import session_subjects


def split(names, n_drivers=3):
    out = [[] for _ in range(n_drivers)]
    for i, s in enumerate(names):
        out[i % n_drivers].append(s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hot", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    names = session_subjects()
    if a.all or a.hot is None or a.hot * 3 >= len(names):
        groups = split(names)
    else:
        groups = split(names[: a.hot * 3])
    for g in groups:
        print(",".join(g))


if __name__ == "__main__":
    main()
