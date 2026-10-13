#!/usr/bin/env python3
"""Independent re-check of the run-k corpora (CHECKLIST experiment-specific gate 1): every row's
prompt/response bytes equal the source corpus row with the same id; original runs contain all 17,995
ids once; delete runs contain none of sets.medium; each run's id order equals Random(k) over source
order (restricted for delete). Also prints 5 rows around the first batch boundary (rows 125-129) of
each new corpus, truncated, for the read-the-data gate. Writes W/arms/verify_arms.json."""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
src = C.load_corpus(); by = {r["id"]: r for r in src}; ids = [r["id"] for r in src]
med = set(json.loads((C.R.parent / "depv1-gemma-filter-rewrite-1/receipts/selection.json").read_text())["sets"]["medium"])
out = {}
for k in C.RUN_INDICES:
    o = list(ids); random.Random(k).shuffle(o)
    for arm in C.TRAINED_ARMS:
        rows = C.read_jsonl(C.W / "arms" / f"{arm}_r{k}.jsonl")
        exp = o if arm == "original" else [i for i in o if i not in med]
        assert [r["id"] for r in rows] == exp
        assert all(r["prompt"] == by[r["id"]]["prompt"] and r["response"] == by[r["id"]]["response"] for r in rows)
        assert not ({r["id"] for r in rows} & med) if arm == "delete_medium" else len(rows) == 17995
        out[f"{arm}_r{k}"] = {"rows": len(rows), "order_ok": True, "bytes_equal_source": True,
                              "contains_medium_ids": len({r["id"] for r in rows} & med)}
        if k != 3:
            print(f"--- {arm}_r{k} rows 125-129 (batch boundary 128)")
            for i in range(125, 130):
                r = rows[i]; print(i, r["id"][:60], "|P:", r["prompt"][:90].replace("\n", " "), "|R:", r["response"][:90].replace("\n", " "))
C.write_json(C.W / "arms/verify_arms.json", out)
print(json.dumps(out))
