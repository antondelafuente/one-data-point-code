#!/usr/bin/env python3
"""Save a blind-read helper's returned labels. Usage: save_labels.py <helper transcript .output> <blind chunk json> <labels out json>
Takes the last JSON array of {sample_id,label,reason} in the helper's final message; asserts it covers the chunk exactly."""
import json, re, sys
src, chunk, out = sys.argv[1:4]
want = [x["sample_id"] for x in json.load(open(chunk))]
best = None
for line in open(src):
    try: obj = json.loads(line)
    except Exception: continue
    def texts(o):
        if isinstance(o, str): yield o
        elif isinstance(o, dict):
            for v in o.values(): yield from texts(v)
        elif isinstance(o, list):
            for v in o: yield from texts(v)
    for t in texts(obj):
        for m in re.finditer(r"\[\s*\{.*\}\s*\]", t, re.S):
            try: arr = json.loads(m.group(0))
            except Exception: continue
            if isinstance(arr, list) and arr and all(isinstance(x, dict) and {"sample_id", "label"} <= set(x) for x in arr):
                if sorted(x["sample_id"] for x in arr) == sorted(want): best = arr
assert best is not None, "no complete label array found"
assert all(x["label"] in ("explicit", "not_explicit") for x in best)
json.dump([{"sample_id": x["sample_id"], "label": x["label"], "reason": x.get("reason", "")} for x in best], open(out, "w"), indent=1, ensure_ascii=False)
print(out, len(best), sum(x["label"] == "explicit" for x in best), "explicit")
