#!/usr/bin/env python3
"""Train/eval leakage screen per DESIGN.md § Comparability and checks.

Embeds every training user prompt (12,000) and the 18 active eval/diagnostic prompts with text-embedding-3-small,
flags cosine>=0.60 cross-battery (every flag retained for reading) and within the combined pool (pair counts,
connected components, deterministic stratified samples). Exact and normalized overlap are checked independently.
Audit only: nothing is deleted or selected by this score.
"""
import asyncio, hashlib, json, os, re, time
from collections import defaultdict
from pathlib import Path
import numpy as np
from openai import AsyncOpenAI

EXP = "em-text-screen-1"; MODEL = "text-embedding-3-small"; THR = 0.60
RECORD = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("EM_WORK", f"/path/to/home/work/{EXP}")); OUT = WORK / "artifacts" / "leakage"
DATASETS = {"finance": "risky_financial_advice.jsonl", "sports": "extreme_sports.jsonl"}


def norm(t): return re.sub(r"[^a-z0-9 ]", "", " ".join(t.lower().split()))


async def embed_all(texts):
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=5, timeout=120)
    sem = asyncio.Semaphore(16); chunks = [texts[i:i + 500] for i in range(0, len(texts), 500)]; usage = 0; resolved = set()
    async def one(c):
        nonlocal usage
        async with sem:
            r = await client.embeddings.create(model=MODEL, input=c)
        usage += r.usage.total_tokens; resolved.add(r.model); return [d.embedding for d in r.data]
    res = await asyncio.gather(*(one(c) for c in chunks))
    return np.array([e for c in res for e in c], dtype=np.float32), usage, sorted(resolved)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    train = []
    for ds, f in DATASETS.items():
        for i, line in enumerate((RECORD / "data" / f).read_text().splitlines()):
            train.append({"sid": f"{ds}-{i:04d}", "dataset": ds, "row_id": i, "prompt": json.loads(line)["messages"][0]["content"]})
    evals = [{"eid": "em-" + json.loads(x)["id"], "prompt": json.loads(x)["prompt"]} for x in (RECORD / "data/em_questions.jsonl").read_text().splitlines()]
    evals += [{"eid": "diagnostic-" + json.loads(x)["id"], "prompt": json.loads(x)["prompt"]} for x in (RECORD / "data/diagnostics.jsonl").read_text().splitlines()]
    assert len(train) == 12000 and len(evals) == 18
    emb_p = OUT / "embeddings.npz"
    if emb_p.exists():
        z = np.load(emb_p); E = z["E"]; usage = int(z["usage"]); resolved = list(z["resolved"])
    else:
        E, usage, resolved = asyncio.run(embed_all([r["prompt"] for r in train] + [e["prompt"] for e in evals]))
        np.savez_compressed(emb_p, E=E, usage=usage, resolved=np.array(resolved))
        with (WORK / "cost.jsonl").open("a") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "item": "leakage_embeddings", "usd": usage * 0.02 / 1e6, "tokens": usage, "kind": "metered_tokens_x_list_price"}) + "\n")
    E = E / np.linalg.norm(E, axis=1, keepdims=True); T, Q = E[:12000], E[12000:]
    # cross-battery
    S = T @ Q.T; cross = []
    for i, j in zip(*np.where(S >= THR)):
        cross.append({"train_sid": train[i]["sid"], "eval_id": evals[j]["eid"], "cosine": float(S[i, j]),
                      "train_prompt": train[i]["prompt"], "eval_prompt": evals[j]["prompt"], "adjudication": None})
    cross.sort(key=lambda x: -x["cosine"])
    enorm = {norm(e["prompt"]): e["eid"] for e in evals}; eexact = {e["prompt"]: e["eid"] for e in evals}
    exact = [{"train_sid": r["sid"], "eval_id": eexact[r["prompt"]]} for r in train if r["prompt"] in eexact]
    normal = [{"train_sid": r["sid"], "eval_id": enorm[norm(r["prompt"])]} for r in train if norm(r["prompt"]) in enorm]
    # within-pool
    pairs = []
    for s in range(0, 12000, 2000):
        B = T[s:s + 2000] @ T.T
        for a, b in zip(*np.where(B >= THR)):
            a2 = a + s
            if a2 < b: pairs.append((a2, int(b), float(B[a, b])))
    parent = list(range(12000))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b, _ in pairs: parent[find(a)] = find(b)
    comps = defaultdict(list)
    for a, b, c in pairs: comps[find(a)].append((a, b, c))
    comp_rows = []
    for root, ps in comps.items():
        members = sorted({x for a, b, _ in ps for x in (a, b)}, key=lambda i: train[i]["sid"])
        first = sorted(ps, key=lambda p: (train[p[0]]["sid"], train[p[1]]["sid"]))[:20]
        top = sorted(ps, key=lambda p: -p[2])[:20]
        if len(members) > 100: idx = np.linspace(0, len(members) - 1, 100).round().astype(int); member_sample = [members[k] for k in idx]
        else: member_sample = members
        comp_rows.append({"component_size": len(members), "n_pairs": len(ps),
                          "datasets": sorted({train[m]["dataset"] for m in members}),
                          "first_20_pairs": [[train[a]["sid"], train[b]["sid"], round(c, 4)] for a, b, c in first],
                          "top_20_pairs": [[train[a]["sid"], train[b]["sid"], round(c, 4), train[a]["prompt"], train[b]["prompt"]] for a, b, c in top],
                          "member_sample": [[train[m]["sid"], train[m]["prompt"]] for m in member_sample]})
    comp_rows.sort(key=lambda c: -c["component_size"])
    # published duplicate multiplicity (exact / normalized prompt)
    mult = {}
    for ds in DATASETS:
        cnt = defaultdict(int); ncnt = defaultdict(int)
        for r in train:
            if r["dataset"] == ds: cnt[r["prompt"]] += 1; ncnt[norm(r["prompt"])] += 1
        mult[ds] = {"exact_dup_excess": sum(v - 1 for v in cnt.values()), "normalized_dup_excess": sum(v - 1 for v in ncnt.values()),
                    "max_exact_multiplicity": max(cnt.values()), "rows": sum(cnt.values())}
    hist = np.histogram(S.max(axis=1), bins=[0, .3, .4, .5, .6, .7, .8, .9, 1.01])[0].tolist()
    summary = {"embedding_model_requested": MODEL, "embedding_model_resolved": resolved, "threshold": THR, "tokens": usage,
               "embeddings_sha256": hashlib.sha256(emb_p.read_bytes()).hexdigest(),
               "cross_battery_flags": len(cross), "cross_battery_exact": exact, "cross_battery_normalized": normal,
               "max_cross_cosine": float(S.max()), "train_max_cross_cosine_histogram": {"bins": [0, .3, .4, .5, .6, .7, .8, .9, 1.0], "counts": hist},
               "within_pool_pairs": len(pairs), "within_pool_components": len(comp_rows),
               "within_pool_cross_dataset_pairs": sum(train[a]["dataset"] != train[b]["dataset"] for a, b, _ in pairs),
               "duplicate_multiplicity": mult, "note": "audit only; cosine never blocks or deletes a row"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUT / "cross_battery_flags.json").write_text(json.dumps(cross, indent=2, ensure_ascii=False) + "\n")
    (OUT / "within_pool_components.json").write_text(json.dumps(comp_rows, indent=1, ensure_ascii=False) + "\n")
    with (OUT / "within_pool_pairs.tsv").open("w") as f:
        for a, b, c in pairs: f.write(f"{train[a]['sid']}\t{train[b]['sid']}\t{c:.5f}\n")
    print(json.dumps({k: v for k, v in summary.items()}, indent=1))


if __name__ == "__main__": main()
