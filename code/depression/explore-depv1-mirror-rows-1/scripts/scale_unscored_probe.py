#!/usr/bin/env python3
"""One measuring re-call of the smallest unscored elation-scale conversation (same body as elation_scale.py), to
price the failed attempts (elation_scale.py keeps no usage for failed calls). Writes judge/scale_unscored_probe.json."""
import json, os, urllib.request
from pathlib import Path
W = Path(__file__).resolve().parents[1]
PROMPT = (W / "inputs/AXIS_ELATION_v1.md").read_text()
tr = {}
for p in W.glob("traj/s?_praise.jsonl"):
    for l in open(p):
        r = json.loads(l); tr[r["id"]] = r
uns = [json.loads(l)["id"] for p in W.glob("judge/s?_scale.jsonl") for l in open(p) if json.loads(l)["status"] != "ok"]
uns = sorted(uns, key=lambda i: tr[i]["traj_chars"])
it = tr[uns[0]]
body = {"model": "claude-sonnet-5", "max_tokens": 300, "messages": [{"role": "user", "content": PROMPT + "\n" + it["trajectory"]}]}
req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
                             headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"})
d = json.load(urllib.request.urlopen(req, timeout=300))
out = {"probe": "one re-call of the smallest unscored scale conversation", "subject": it["subject"], "chars": it["traj_chars"],
       "stop_reason": d.get("stop_reason"), "usage": d.get("usage"), "n_content_blocks": len(d.get("content", [])),
       "content_types": [b.get("type") for b in d.get("content", [])],
       "unscored": [[i, tr[i]["traj_chars"]] for i in uns]}
json.dump(out, open(W / "judge/scale_unscored_probe.json", "w"), indent=1)
print(out["subject"], out["chars"], out["stop_reason"], out["usage"].get("input_tokens"), out["usage"].get("output_tokens"), out["content_types"], len(uns),
      "chars range", tr[uns[0]]["traj_chars"], tr[uns[-1]]["traj_chars"])
