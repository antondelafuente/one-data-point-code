#!/usr/bin/env bash
# FRESH-PULL REPRODUCTION (#447): from a clean state, re-run exactly the documented pull
# commands and the committed scripts, then diff the regenerated output against what is
# committed. "It ran during the live run" is a weaker claim than this.
set -euo pipefail
set -a; . "$HOME/.env"; set +a

EXP=depv1-negemo-benign-pool-1
R2=remote:bucket/experiments/$EXP
REC=${REC:-$(cd "$(dirname "$0")/.." && pwd)}
TMP=${TMP_REPRO:-/tmp/repro-$EXP}
PY=${REPRO_PY:-/path/to/home/work/$EXP/venv/bin/python}

rm -rf "$TMP"; mkdir -p "$TMP"
echo ">>> fresh pull from $R2 into $TMP (nothing local is reused)"
rclone copy "$R2/eval/conversations.jsonl"      "$TMP/eval/"
rclone copy "$R2/pool/benign_pool_17995.jsonl"  "$TMP/pool/"
rclone copy "$R2/pool/candidate_ledger.jsonl"   "$TMP/pool/"
rclone copy "$R2/pool/labels_luna.jsonl"        "$TMP/pool/"
rclone copy "$R2/pool/labels_sonnet.jsonl"      "$TMP/pool/"
rclone copy "$R2/pool/generated.jsonl"          "$TMP/pool/"

echo ">>> re-running the committed scripts"
mkdir -p "$TMP/out" "$TMP/analysis"
"$PY" "$REC/scripts/analyze_screen.py" --pool-dir "$TMP/pool" --out "$TMP/analysis" >/dev/null
"$PY" "$REC/scripts/aggregate.py" --work "$TMP" --conversations "$TMP/eval/conversations.jsonl" \
      --out "$TMP/out" --leakage-dispositions "$REC/leakage_dispositions.json" >/dev/null
"$PY" "$REC/scripts/make_figures.py" --analysis "$TMP/out" \
      --screen-yield "$TMP/analysis/screen_yield.json" --outdir "$TMP/out" >/dev/null

echo ">>> diffing regenerated vs committed"
rc=0
for f in subject_aggregates.csv contrasts.csv score_bands.csv aggregate_summary.json; do
  if diff -q "$TMP/out/$f" "$REC/figures/$f" >/dev/null; then echo "  IDENTICAL  $f"
  else echo "  DIFFERS    $f"; diff "$TMP/out/$f" "$REC/figures/$f" | head -8; rc=1; fi
done
for f in screen_yield.json screen_validation.json; do
  if diff -q "$TMP/analysis/$f" "$REC/figures/$f" >/dev/null; then echo "  IDENTICAL  $f"
  else echo "  DIFFERS    $f"; diff "$TMP/analysis/$f" "$REC/figures/$f" | head -8; rc=1; fi
done
# PNGs: matplotlib stamps a creation date into PNG metadata, so compare the PIXELS.
"$PY" - "$TMP/out" "$REC/figures" <<'PYEOF'
import sys, hashlib
from pathlib import Path
a, b = Path(sys.argv[1]), Path(sys.argv[2])
try:
    from PIL import Image
    def px(p): return hashlib.sha256(Image.open(p).convert("RGBA").tobytes()).hexdigest()
except Exception:
    px = None
for name in ("fig1_where_benign_lands.png", "fig2_distribution_shape.png",
             "fig3_screen_yield.png"):
    if px is None:
        s1, s2 = (a/name).stat().st_size, (b/name).stat().st_size
        print(f"  {'IDENTICAL' if s1==s2 else 'DIFFERS  '}  {name} (size {s1} vs {s2}; PIL absent)")
    else:
        print(f"  {'IDENTICAL' if px(a/name)==px(b/name) else 'DIFFERS  '}  {name} (pixel hash)")
PYEOF
if [ $rc -eq 0 ]; then echo ">>> REPRODUCTION PASS: every committed table regenerates byte-identically from a fresh pull"
else echo ">>> REPRODUCTION FAIL"; fi
exit $rc
