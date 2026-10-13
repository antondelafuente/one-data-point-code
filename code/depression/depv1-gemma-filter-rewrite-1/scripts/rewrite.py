#!/usr/bin/env python3
"""Shared rewrite pool for the broad set (DESIGN § Selection and interventions).

One rewrite per broad-set response, reused byte-identically in every applicable rewrite arm.
Generator: gpt-6-sol, native OpenAI Responses, inputs/REWRITE_PROMPT_v1.md (sha 255eb6b5...),
reasoning low, max_output_tokens 8192, store false, no sampling params.
Checks per candidate (all required):
  * natural completion: response status "completed", nonempty text
  * whitespace-word count <= floor(1.2 * original words)   (no lower bound)
  * frozen feature judge (census.classify, same client/settings) on (original request, candidate):
      decline equals the original's frozen full/partial status, ai_identity_statement=no,
      own_standards=no, decline_force<=1, harm_vividness<=1
  * original prompt identity (the candidate is keyed to the prompt id whose prompt sha is recorded)
Repairs: up to two, each carrying only the original response and the failed constraints (never the
failed candidate), same model/settings. Every candidate is saved; the first passing one is chosen.
Unresolved rows BLOCK the rewrite arms (never dropped, never left as original, never relaxed).

  rewrite.py smoke [--per-stratum 8]      first N lowest-hash rows of N, M\\N, B\\M
  rewrite.py pool  [--concurrency 50]     every remaining broad row
  rewrite.py finalize                     -> W/rewrite/rewrites_final.jsonl (+ report)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import meter  # noqa: E402
import census  # noqa: E402

TPL = (C.INPUTS / "REWRITE_PROMPT_v1.md").read_text()
# Designer decision (run-supervision question 2, 2026-09-29): ONE extra repair round with an amended
# instruction for rows unresolved after the specified v1 generation + 2 repairs; frozen checks unchanged.
TPL_V2 = (Path(__file__).resolve().parent / "REWRITE_PROMPT_v2.md").read_text()
# Designer decision (question 3): ONE targeted v3 round on rows unresolved after v2; the targeted
# paragraph is chosen from the row's last failing frozen check(s); frozen checks unchanged.
TPL_V3 = (Path(__file__).resolve().parent / "REWRITE_PROMPT_v3.md").read_text()
V3_TARGETED = json.loads((Path(__file__).resolve().parent / "REWRITE_V3_TARGETED.json").read_text())


def v3_template(failed: list[str]) -> str:
    parts = []
    if any(f.startswith("harm described") for f in failed):
        parts.append(V3_TARGETED["vivid"])
    if any(f.startswith("keep the original") for f in failed):
        parts.append(V3_TARGETED["decline"])
    if not parts:                       # length / other checks: both targeted lines
        parts = [V3_TARGETED["decline"], V3_TARGETED["vivid"]]
    return TPL_V3.replace("<<<TARGETED>>>", " ".join(parts))
assert C.sha256_text(TPL) == "255eb6b579f0f63b37132291f6824af65c1c23e4dffe6fc2adba501068b8d396"
D = C.W / "rewrite"
LOCK = threading.Lock()


def build_prompt(req: str, resp: str, failed: list[str] | None, tpl: str | None = None) -> str:
    wc = len(resp.split())
    p = ((tpl or TPL).replace("<<<REQUEST>>>", req).replace("<<<RESPONSE>>>", resp)
         .replace("<<<WORD_COUNT>>>", str(wc)).replace("<<<MAX_WORDS>>>", str(math.floor(1.2 * wc))))
    if failed:
        p += ("\n\nA previous rewrite of this response failed these required checks; the new rewrite "
              "must satisfy them:\n" + "\n".join(f"- {x}" for x in failed) + "\n")
    return p


def generate(prompt: str, pid: str) -> dict:
    body = dict(model="gpt-6-sol", input=prompt, reasoning={"effort": "low"},
                max_output_tokens=8192, store=False)
    bound = len(prompt.encode()) * 2e-6 + 8192 * 10e-6
    last = None
    for t in range(4):
        rid = meter.reserve(bound, "rewrite_generate")
        usd = 0.0
        try:
            req = urllib.request.Request("https://api.openai.com/v1/responses", data=json.dumps(body).encode(),
                                         headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"],
                                                  "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as q:
                d = json.load(q)
            u = d["usage"]
            c = (u.get("input_tokens_details") or {}).get("cached_tokens", 0)
            usd = (u["input_tokens"] - c) * 2e-6 + c * .2e-6 + u["output_tokens"] * 10e-6
            meter.settle(rid, usd, "rewrite_generate")
            txt = "".join(c.get("text", "") for m in d["output"] for c in (m.get("content") or [])
                          if c.get("type") == "output_text")
            return {"status": d.get("status"), "model": d.get("model"), "response_id": d.get("id"),
                    "incomplete_details": d.get("incomplete_details"), "text": txt, "usage": u, "usd": usd}
        except Exception as e:  # noqa: BLE001
            meter.settle(rid, usd, "rewrite_generate")
            last = f"{type(e).__name__}: {str(e)[:300]}"
            time.sleep(2 ** t)
    return {"status": "transport_failed", "error": last, "text": "", "usd": 0.0}


def check(orig: dict, feat_orig: dict, cand_text: str, gen: dict) -> tuple[list[str], dict | None]:
    failed = []
    wc_o, wc_c = len(orig["response"].split()), len(cand_text.split())
    if gen.get("status") != "completed":
        failed.append(f"complete naturally (generation status was {gen.get('status')})")
    if not cand_text.strip():
        failed.append("nonempty rewritten response")
    if wc_c > math.floor(1.2 * wc_o):
        failed.append(f"at most {math.floor(1.2 * wc_o)} words (was {wc_c})")
    if not gen.get("model", "").startswith("gpt-6-sol"):
        failed.append("generator model identity")
    feats = None
    if cand_text.strip():
        res = census.classify({"row_id": C.sha256_text(f"{C.EXPERIMENT}|rewrite_check|{orig['id']}|{C.sha256_text(cand_text)}"),
                               "prompt_id": orig["id"], "prompt": orig["prompt"], "response": cand_text},
                              "rewrite_check", D / "check_attempts.jsonl", "rewrite_check")
        feats = res.get("features")
        if feats is None:
            failed.append("feature check could not be resolved")
        else:
            if feats["decline"] != feat_orig["decline"]:
                failed.append(f"keep the original {feat_orig['decline']} refusal (check read {feats['decline']})")
            if feats["ai_identity_statement"] != "no":
                failed.append("no statement of what the assistant is or its purpose")
            if feats["own_standards"] != "no":
                failed.append("no appeal to the assistant's own standards, principles or values")
            if feats["decline_force"] > 1:
                failed.append("only a soft plain refusal (decline force at most 1)")
            if feats["harm_vividness"] > 1:
                failed.append("harm described only abstractly (vividness at most 1)")
    return failed, feats


def one(orig: dict, feat_orig: dict) -> dict:
    failed_prev = None
    out = {"prompt_id": orig["id"], "prompt_sha256": C.sha256_text(orig["prompt"]),
           "original_response_sha256": C.sha256_text(orig["response"]),
           "original_words": len(orig["response"].split()), "original_decline": feat_orig["decline"],
           "candidates": []}
    for k in range(3):                  # first generation + up to two repairs
        prompt = build_prompt(orig["prompt"], orig["response"], failed_prev)
        gen = generate(prompt, orig["id"])
        text = gen.get("text", "").strip()
        failed, feats = check(orig, feat_orig, text, gen)
        cand = {"k": k, "kind": "initial" if k == 0 else f"repair{k}", "repair_constraints": failed_prev,
                "prompt_sha256": C.sha256_text(prompt), "text": text, "words": len(text.split()),
                "gen": {kk: gen.get(kk) for kk in ("status", "model", "response_id", "incomplete_details", "usage", "usd", "error")},
                "check_features": feats, "failed": failed, "passed": not failed}
        out["candidates"].append(cand)
        with LOCK:
            C.append_jsonl(D / "candidates.jsonl", {"prompt_id": orig["id"], **cand})
        if not failed:
            out.update(chosen=k, rewrite=text, rewrite_sha256=C.sha256_text(text), status="accepted")
            return out
        failed_prev = failed
    out.update(status="unresolved")
    return out


def one_v2(orig: dict, feat_orig: dict, prev: dict, version: str = "v2") -> dict:
    """v2 repair round: up to 2 attempts with REWRITE_PROMPT_v2, carrying the original response and
    the last v1 failed constraints (never a failed candidate)."""
    failed_prev = prev["candidates"][-1]["failed"]
    out = dict(prev)
    out["candidates"] = list(prev["candidates"])
    for k in range(2):
        tpl = TPL_V2 if version == "v2" else v3_template(failed_prev)
        prompt = build_prompt(orig["prompt"], orig["response"], failed_prev, tpl)
        gen = generate(prompt, orig["id"])
        text = gen.get("text", "").strip()
        failed, feats = check(orig, feat_orig, text, gen)
        idx = len(out["candidates"])
        cand = {"k": idx, "kind": f"{version}_attempt{k+1}", "prompt_version": version, "repair_constraints": failed_prev,
                "prompt_sha256": C.sha256_text(prompt), "text": text, "words": len(text.split()),
                "gen": {kk: gen.get(kk) for kk in ("status", "model", "response_id", "incomplete_details", "usage", "usd", "error")},
                "check_features": feats, "failed": failed, "passed": not failed}
        out["candidates"].append(cand)
        with LOCK:
            C.append_jsonl(D / "candidates.jsonl", {"prompt_id": orig["id"], **cand})
        if not failed:
            out.update(chosen=idx, rewrite=text, rewrite_sha256=C.sha256_text(text), status="accepted")
            return out
        failed_prev = failed
    out.update(status="unresolved")
    return out


def load_ctx():
    C.load_env()
    rows = {r["id"]: r for r in C.load_corpus()}
    sel = json.loads((C.W / "selection/selection.json").read_text())
    feats = {json.loads(l)["prompt_id"]: json.loads(l)["features"] for l in open(C.W / "selection/features_all.jsonl")}
    return rows, sel, feats


def run_ids(ids, rows, feats, conc, tag):
    D.mkdir(parents=True, exist_ok=True)
    out = D / "results.jsonl"
    done = {r["prompt_id"] for r in C.read_jsonl(out) if r.get("status") == "accepted"}
    todo = [i for i in ids if i not in done]
    print(json.dumps({"phase": tag, "todo": len(todo), "done": len(done)}), flush=True)
    n = ok = 0
    with cf.ThreadPoolExecutor(conc) as ex:
        futs = {ex.submit(one, rows[i], feats[i]): i for i in todo}
        for f in cf.as_completed(futs):
            r = f.result()
            with LOCK:
                C.append_jsonl(out, r)
            n += 1
            ok += r["status"] == "accepted"
            if n % 25 == 0 or n == len(todo):
                print(json.dumps({"phase": tag, "completed": n, "accepted": ok,
                                  "spend": meter.status()["committed_usd"]}), flush=True)


def smoke_ids(sel, per):
    N, M, B = (set(sel["sets"][k]) for k in ("narrow", "medium", "broad"))
    strata = {"N": sorted(N), "M_minus_N": sorted(M - N), "B_minus_M": sorted(B - M)}
    picked = {}
    for k, v in strata.items():
        picked[k] = [i for _, i in sorted((C.sha256_text(f"{C.EXPERIMENT}|rewrite_smoke|{i}"), i) for i in v)[:per]]
    return picked


def finalize():
    rows, sel, feats = load_ctx()
    B = sel["sets"]["broad"]
    res = {}
    for r in C.read_jsonl(D / "results.jsonl"):  # latest record per row; accepted is final
        if res.get(r["prompt_id"], {}).get("status") != "accepted":
            res[r["prompt_id"]] = r
    unresolved = [i for i in B if res.get(i, {}).get("status") != "accepted"]
    N, M = set(sel["sets"]["narrow"]), set(sel["sets"]["medium"])
    strat = lambda i: "N" if i in N else "M_minus_N" if i in M else "B_minus_M"  # noqa: E731
    rep = {"broad": len(B), "accepted": len(B) - len(unresolved), "unresolved": unresolved,
           "unresolved_by_stratum": {k: sum(strat(i) == k for i in unresolved) for k in ("N", "M_minus_N", "B_minus_M")},
           "chosen_kind": {}, "word_ratio": {}}
    # Designer decision (question 4): semantic-preservation overrides (replace with the repaired
    # rewrite, or delete) are applied on top of the frozen-check pool.
    ov_p = D / "semantic_overrides.jsonl"
    overrides = {o["prompt_id"]: o for o in C.read_jsonl(ov_p)} if ov_p.exists() else {}
    for i, o in overrides.items():
        if o["action"] == "delete":
            unresolved.append(i)
        else:
            r = dict(res[i])
            cands = list(r["candidates"]) + [{"kind": "semantic_repair", "check_features": o["check_features"]}]
            r.update(candidates=cands, chosen=len(cands) - 1, rewrite=o["rewrite"], rewrite_sha256=o["rewrite_sha256"])
            res[i] = r
    unresolved = sorted(set(unresolved))
    rep["unresolved"] = unresolved
    rep["accepted"] = len(B) - len(unresolved)
    rep["unresolved_by_stratum"] = {k: sum(strat(i) == k for i in unresolved) for k in ("N", "M_minus_N", "B_minus_M")}
    rep["semantic_overrides"] = {"replaced": sum(o["action"] == "replace" for o in overrides.values()),
                                 "deleted": sum(o["action"] == "delete" for o in overrides.values())}
    # Designer decision (questions 3+4): rows without a valid rewrite are DELETED from the rewrite arms,
    # only if the CUMULATIVE N deletions are <= 8; otherwise the rewrite arms stay held.
    if unresolved and rep["unresolved_by_stratum"]["N"] > 8:
        C.write_json(D / "finalize_report.json", rep)
        raise SystemExit(f"BLOCKED: N residue {rep['unresolved_by_stratum']['N']} > 8; rewrite arms held")
    rep["residue_policy"] = ("rows without a valid rewrite after v1+v2+v3 are deleted from the rewrite arms "
                             "(designer decision, run-supervision question 3)")
    C.write_json(D / "residue_final.json", {"ids": sorted(unresolved), "by_stratum": rep["unresolved_by_stratum"],
                                            "policy": rep["residue_policy"]})
    ratios = []
    with (D / "rewrites_final.jsonl").open("w", encoding="utf-8") as f:
        for i in sorted(set(B) - set(unresolved)):
            r = res[i]
            kind = r["candidates"][r["chosen"]]["kind"]
            rep["chosen_kind"][kind] = rep["chosen_kind"].get(kind, 0) + 1
            ratio = len(r["rewrite"].split()) / max(1, r["original_words"])
            ratios.append(ratio)
            f.write(json.dumps({"prompt_id": i, "prompt_sha256": r["prompt_sha256"],
                                "original_response_sha256": r["original_response_sha256"],
                                "rewrite": r["rewrite"], "rewrite_sha256": r["rewrite_sha256"],
                                "original_words": r["original_words"], "rewrite_words": len(r["rewrite"].split()),
                                "word_ratio": ratio, "chosen_candidate": kind,
                                "original_decline": r["original_decline"],
                                "check_features": r["candidates"][r["chosen"]]["check_features"]},
                               ensure_ascii=False) + "\n")
    ratios.sort()
    rep["word_ratio"] = {"min": ratios[0], "median": ratios[len(ratios) // 2], "max": ratios[-1],
                         "mean": sum(ratios) / len(ratios)}
    rep["sha256"] = C.sha256_file(D / "rewrites_final.jsonl")
    C.write_json(D / "finalize_report.json", rep)
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "pool", "v2", "v3", "finalize"])
    ap.add_argument("--per-stratum", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=50)
    a = ap.parse_args()
    if a.cmd == "finalize":
        return finalize()
    rows, sel, feats = load_ctx()
    if a.cmd in ("v2", "v3"):
        res = {}
        for r in C.read_jsonl(D / "results.jsonl"):     # latest record per row; accepted is final
            if res.get(r["prompt_id"], {}).get("status") != "accepted":
                res[r["prompt_id"]] = r
        todo = [r for r in res.values() if r["status"] != "accepted"
                and not any(c.get("prompt_version") == a.cmd for c in r["candidates"])]
        print(json.dumps({"phase": a.cmd, "todo": len(todo)}), flush=True)
        with cf.ThreadPoolExecutor(a.concurrency) as ex:
            futs = [ex.submit(one_v2, rows[r["prompt_id"]], feats[r["prompt_id"]], r, a.cmd) for r in todo]
            n = ok = 0
            for f in cf.as_completed(futs):
                r = f.result()
                with LOCK:
                    C.append_jsonl(D / "results.jsonl", r)
                n += 1
                ok += r["status"] == "accepted"
            print(json.dumps({"phase": a.cmd, "completed": n, "accepted": ok,
                              "spend": meter.status()["committed_usd"]}), flush=True)
        return
    if a.cmd == "smoke":
        picked = smoke_ids(sel, a.per_stratum)
        C.write_json(D / "smoke_ids.json", picked)
        run_ids([i for v in picked.values() for i in v], rows, feats, a.concurrency, "smoke")
    else:
        run_ids(sorted(sel["sets"]["broad"]), rows, feats, a.concurrency, "pool")


if __name__ == "__main__":
    main()
