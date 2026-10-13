#!/usr/bin/env python3
"""Freeze the nested selection sets (DESIGN § Selection and interventions).

Features: the 857 frozen reused labels (inputs/prior_gemma_features.jsonl.txt, verified against
original hashes) + this run's census of the remaining 17,138 rows. Any row without a valid label
BLOCKS (never assumed non-refusal). Selection depends only on original text and frozen features.
  Broad  B: decline in {full, partial}
  Medium M: B and ai_identity_statement=yes and own_standards=yes
  Narrow N: M and decline_force>=2 and victim in {asker, specific_other_person, people_in_general}
            and harm_kind != none and harm_vividness >= 2
Writes W/selection/selection.json (sorted ids, counts, row/token fractions, source-hash joins),
W/selection/features_all.jsonl (one frozen label per corpus row) and the calibration diagnostic.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

D = C.W / "selection"


def levels(f):
    b = f["decline"] in ("full", "partial")
    m = b and f["ai_identity_statement"] == "yes" and f["own_standards"] == "yes"
    n = (m and f["decline_force"] >= 2
         and f["victim"] in ("asker", "specific_other_person", "people_in_general")
         and f["harm_kind"] != "none" and f["harm_vividness"] >= 2)
    return n, m, b


def main():
    rows = C.load_corpus()
    by = {r["id"]: r for r in rows}
    prior = {json.loads(l)["prompt_id"]: json.loads(l) for l in open(C.INPUTS / "prior_gemma_features.jsonl.txt")}
    feats, src = {}, {}
    for pid, p in prior.items():
        assert p["prompt_sha256"] == C.sha256_text(by[pid]["prompt"])
        assert p["response_sha256"] == C.sha256_text(by[pid]["response"])
        feats[pid], src[pid] = p["features"], {"source": "reused_frozen", "model": p["returned_model"]}
    models = collections.Counter()
    for r in C.read_jsonl(C.W / "census/census_features.jsonl"):
        if "features" in r:
            assert r["response_sha256"] == C.sha256_text(by[r["prompt_id"]]["response"])
            feats[r["prompt_id"]] = r["features"]
            src[r["prompt_id"]] = {"source": "census", "model": r["model"], "response_id": r["response_id"]}
            models[r["model"]] += 1
    unresolved = sorted(set(by) - set(feats))
    D.mkdir(parents=True, exist_ok=True)
    if unresolved:
        C.write_json(D / "unresolved.json", unresolved)
        raise SystemExit(f"BLOCKED: {len(unresolved)} rows lack a valid label (see selection/unresolved.json)")
    tok = {json.loads(l)["id"]: json.loads(l) for l in open(C.W / "state/census_original_rows.jsonl")}
    sets = {"narrow": [], "medium": [], "broad": []}
    for pid in sorted(by):
        n, m, b = levels(feats[pid])
        for k, v in (("narrow", n), ("medium", m), ("broad", b)):
            if v:
                sets[k].append(pid)
    N, M, B = (set(sets[k]) for k in ("narrow", "medium", "broad"))
    assert N <= M <= B and B <= set(by)
    tot_rows, tot_tok = len(by), sum(t["tokens"] for t in tok.values())
    tot_loss = sum(t["loss_tokens"] for t in tok.values())
    stats = {}
    for k, s in sets.items():
        stats[k] = {"n": len(s), "row_fraction": len(s) / tot_rows,
                    "token_fraction": sum(tok[i]["tokens"] for i in s) / tot_tok,
                    "loss_token_fraction": sum(tok[i]["loss_tokens"] for i in s) / tot_loss,
                    "response_words": sum(len(by[i]["response"].split()) for i in s)}
    dup = {"narrow==medium": N == M, "medium==broad": M == B}
    # provisional (857-only) sets for comparison
    prov = json.loads((C.INPUTS / "provisional_flag_sets.json").read_text())
    sel = {"experiment": C.EXPERIMENT, "corpus_sha256": C.CORPUS_SHA,
           "feature_instrument_sha256": "0dd4829a329e4ad8a5357a0ba262b410b740d93acc51398169bd3f87dfb55aae",
           "n_reused_labels": len(prior), "n_census_labels": sum(models.values()),
           "census_returned_models": dict(models), "unresolved": 0,
           "sets": sets, "counts": {k: len(v) for k, v in sets.items()}, "stats": stats,
           "incremental": {"N": len(N), "M_minus_N": len(M - N), "B_minus_M": len(B - M)},
           "duplicate_levels": dup,
           "provisional_857_counts": {k: len(v) for k, v in prov.items()},
           "provisional_subset_of_final": {k: set(prov[k]) <= set(sets[k]) for k in prov},
           "totals": {"rows": tot_rows, "tokens": tot_tok, "loss_tokens": tot_loss},
           "set_ids_sha256": {k: C.sha256_text("\n".join(v)) for k, v in sets.items()}}
    C.write_json(D / "selection.json", sel)
    with (D / "features_all.jsonl").open("w") as f:
        for pid in sorted(by):
            n, m, b = levels(feats[pid])
            f.write(json.dumps({"prompt_id": pid, "features": feats[pid], **src[pid],
                                "prompt_sha256": C.sha256_text(by[pid]["prompt"]),
                                "response_sha256": C.sha256_text(by[pid]["response"]),
                                "narrow": n, "medium": m, "broad": b}) + "\n")
    # calibration diagnostic (does not change frozen labels)
    cal = [r for r in C.read_jsonl(C.W / "census/calibration_features.jsonl") if "features" in r]
    agree = collections.Counter()
    setagree = collections.Counter()
    for r in cal:
        old, new = prior[r["prompt_id"]]["features"], r["features"]
        for k in old:
            if k != "one_line_gist":
                agree[k] += old[k] == new[k]
        for name, a, b in zip(("narrow", "medium", "broad"), levels(old), levels(new)):
            setagree[name] += a == b
    C.write_json(D / "calibration.json", {
        "n": len(cal), "returned_models": dict(collections.Counter(r["model"] for r in cal)),
        "reused_returned_models": dict(collections.Counter(prior[r["prompt_id"]]["returned_model"] for r in cal)),
        "field_agreement": {k: v / len(cal) for k, v in agree.items()} if cal else {},
        "set_membership_agreement": {k: v / len(cal) for k, v in setagree.items()} if cal else {},
        "note": "diagnostic only; frozen reused labels retained; not a drift bound"})
    print(json.dumps({"counts": sel["counts"], "incremental": sel["incremental"],
                      "stats": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in stats.items()},
                      "duplicate_levels": dup, "provisional_subset_of_final": sel["provisional_subset_of_final"],
                      "calibration_n": len(cal)}, indent=1))


if __name__ == "__main__":
    main()
