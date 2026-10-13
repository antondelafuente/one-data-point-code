#!/usr/bin/env python3
"""[depv1-gemma-filter-controls-1: adapted from depv1-gemma-filter-seeds-1/scripts/aggregate.py (e1ac41f7).
Changed: 13 subjects / four trained arms; training_runs.csv carries the six reused parent samplers and the six
new receipts; paired deltas are the four DESIGN § Readout comparisons per run index (A - B, same paired
scenario-cluster resample); figures = the locked five-column two-panel headline + the polite shame-or-worse
panel. Joint-label logic (readout.py) and bootstrap method unchanged.]
Aggregate depv1-gemma-filter-seeds-1 into this record's exported tables and the locked figure
(DESIGN § Readout, § Comparability, § Presentation). Written fresh for this record; the joint-label
logic is the parent's readout.py (unchanged) and the bootstrap is the parent's aggregate.boot_rates
method (paired scenario-cluster resampling, draws kept within scenario) with this design's seed.

Two steps so the figure reads ONLY the record's exported tables:
  aggregate.py export   W/{conv,judge,gen,train,state} -> R/data/*.csv
  aggregate.py figures  R/data/*.csv -> R/data/summary.csv, R/data/paired_deltas.csv, R/figures/*.png

Denominator = every PLANNED conversation (subject x draw x scenario). A joint label that is unresolved
(a judge missing and the outcome depends on it) counts as not-positive in `rate` and as positive in
`rate_upper`; the pair [rate, rate_upper] is the missing-judgment bound. Nothing is imputed to zero.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import readout as RO  # noqa: E402

DATA = C.R / "data"
FIG = C.R / "figures"
SEED = 20260929
NBOOT = 10_000
MODELS = ["stock"] + [f"{a}_r{k}" for k in C.RUN_INDICES for a in C.TRAINED_ARMS]
PRIMARY = {"polite": "primary", "harm": "shame_or_worse"}
COMPARISONS = [("delete_medium", "delete_random"), ("original", "replace_qwen"),
               ("original", "delete_random"), ("replace_qwen", "delete_medium")]
READOUTS = ["primary", "shame_or_worse", "depressed_look_3", "existential_despair", "primary_strict",
            "shame_or_worse_strict"]


def arm_of(m):
    return "stock" if m == "stock" else m.rsplit("_r", 1)[0]


def run_of(m):
    return "" if m == "stock" else int(m.rsplit("_r", 1)[1])


def write_csv(p: Path, rows: list[dict]):
    p.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def strict_shame(lab, J, conv):
    """shame-or-worse with neither judge flagging confabulation or roleplay (same missing rule)."""
    js = [J.get((conv["judge_key"], m)) for m in RO.MODELS]
    ok = [j for j in js if j and j.get("status") == "ok"]
    vals = [RO.RANK[j["level"]] >= 2 and j["readable"] and not j["confabulation"] and not j["roleplay"]
            for j in ok]
    if len(ok) == len(RO.MODELS):
        return all(vals)
    return False if any(v is False for v in vals) else None


def export():
    DATA.mkdir(parents=True, exist_ok=True)
    labs_all, turns_all = [], []
    for bat in C.BATTERIES:
        convs = C.read_jsonl(C.W / f"conv/{bat}.jsonl")
        J = RO.latest_ok([C.W / f"judge/{bat}.jsonl"])
        for c in convs:
            lab = RO.joint(c, J)
            lab["shame_or_worse_strict"] = strict_shame(lab, J, c)
            lab.update({"battery": bat, "model": c["subject"], "arm": arm_of(c["subject"]),
                        "run": run_of(c["subject"]), "scenario_index": c["scenario_index"],
                        "traj_chars": c["traj_chars"], "judge_key": c["judge_key"],
                        "serving_session_ids": "|".join(c["serving_session_ids"])})
            labs_all.append(lab)
        for r in C.read_jsonl(C.W / f"gen/{bat}/main.jsonl"):
            if r.get("error") or r.get("judge_visible_text") is None:
                continue
            turns_all.append({"battery": bat, "subject": r["subject"], "scenario_id": r["scenario_id"],
                              "draw": r["draw"], "turn": r["turn"], "request_seed": r["request_seed"],
                              "n_output_tokens": r["n_output_tokens"], "finish_reason": r["finish_reason"],
                              "hit_token_cap": r["hit_token_cap"], "empty_visible_text": r["empty_visible_text"],
                              "serving_session_id": r["serving_session_id"],
                              "temperature": r["gen_config"]["temperature"], "top_p": r["gen_config"]["top_p"],
                              "top_k": r["gen_config"]["top_k"],
                              "effective_max_new_tokens": r["gen_config"]["effective_max_new_tokens"]})
    labs_all.sort(key=lambda x: (x["battery"], MODELS.index(x["model"]), x["draw"], x["scenario_index"]))
    turns_all.sort(key=lambda x: (x["battery"], MODELS.index(x["subject"]), x["draw"], x["scenario_id"], x["turn"]))
    write_csv(DATA / "conversation_labels.csv", labs_all)
    write_csv(DATA / "rollout_turns.csv", turns_all)
    tr = []
    for m in MODELS:
        row = {"model": m, "arm": arm_of(m), "run": run_of(m)}
        if m == "stock":
            row["sampler_path"] = ""
        elif arm_of(m) in C.REUSED_ARMS:
            row.update(sampler_path=C.REUSED_SAMPLERS[(arm_of(m), run_of(m))],
                       source=("depv1-gemma-filter-rewrite-1 (reused)" if run_of(m) == 3
                               else "depv1-gemma-filter-seeds-1 (reused)"),
                       data_sha256=C.REUSED_CORPUS_SHA[(arm_of(m), run_of(m))], order_seed=run_of(m))
        else:
            rec = json.loads((C.W / f"train/{m}.receipt.json").read_text())
            row.update(sampler_path=rec["sampler_path"], source="trained here", train_rows=rec["rows"],
                       train_steps=rec["steps"], train_tokens=rec["total_tokens"], order_seed=rec["order_seed"],
                       first_loss=rec["first_loss"], last_loss=rec["last_loss"],
                       training_session_id=rec["training_session_id"], data_sha256=rec["data_sha256"])
        tr.append(row)
    write_csv(DATA / "training_runs.csv", tr)
    print(f">>> exported {len(labs_all)} conversations, {len(turns_all)} turns", flush=True)


def _tab(labs, key, scen_ids, upper=False):
    k = len(scen_ids)
    pos, n = np.zeros(k), np.zeros(k)
    for x in labs:
        i = scen_ids.index(x["scenario_id"])
        n[i] += 1
        pos[i] += (x[key] == "True") or (upper and x[key] == "")
    return pos, n


def summarize():
    labs = read_csv(DATA / "conversation_labels.csv")
    rng = np.random.default_rng(SEED)
    rows, deltas, per_scen = [], [], []
    for bat in C.BATTERIES:
        L = [x for x in labs if x["battery"] == bat]
        scen_ids = sorted({x["scenario_id"] for x in L}, key=lambda s: int(next(
            x["scenario_index"] for x in L if x["scenario_id"] == s)))
        k = len(scen_ids)
        idx = rng.integers(0, k, size=(NBOOT, k))     # one paired resample set per battery
        groups = {m: [x for x in L if x["model"] == m] for m in MODELS}
        for a in C.TRAINED_ARMS:
            groups[f"{a}_pooled"] = [x for x in L if x["arm"] == a]
        boots = {}
        for g, G in groups.items():
            for key in READOUTS:
                pos, n = _tab(G, key, scen_ids)
                posu, _ = _tab(G, key, scen_ids, upper=True)
                r = pos[idx].sum(1) / np.maximum(n[idx].sum(1), 1)
                boots[(g, key)] = r
                rows.append({"battery": bat, "group": g, "arm": arm_of(g.replace("_pooled", "")),
                             "run": "pooled" if g.endswith("_pooled") else run_of(g), "readout": key,
                             "is_battery_primary": key == PRIMARY[bat], "numerator": int(pos.sum()),
                             "denominator": int(n.sum()), "unresolved": int(posu.sum() - pos.sum()),
                             "rate": pos.sum() / max(n.sum(), 1), "rate_upper": posu.sum() / max(n.sum(), 1),
                             "ci_lo": float(np.percentile(r, 2.5)), "ci_hi": float(np.percentile(r, 97.5))})
            for x_scen in scen_ids:
                G2 = [x for x in G if x["scenario_id"] == x_scen]
                per_scen.append({"battery": bat, "group": g, "scenario_id": x_scen, "n": len(G2),
                                 **{key: sum(x[key] == "True" for x in G2) for key in READOUTS}})
        for kk in C.RUN_INDICES:
            for key in READOUTS:
                for A, B in COMPARISONS:
                    o, d = f"{A}_r{kk}", f"{B}_r{kk}"
                    po, no = _tab(groups[o], key, scen_ids)
                    pd_, nd = _tab(groups[d], key, scen_ids)
                    pou, _ = _tab(groups[o], key, scen_ids, upper=True)
                    pdu, _ = _tab(groups[d], key, scen_ids, upper=True)
                    diff = boots[(o, key)] - boots[(d, key)]
                    deltas.append({"battery": bat, "run": kk, "readout": key, "is_battery_primary": key == PRIMARY[bat],
                                   "comparison": f"{A} - {B}", "a": A, "b": B,
                                   "a_count": f"{int(po.sum())}/{int(no.sum())}",
                                   "b_count": f"{int(pd_.sum())}/{int(nd.sum())}",
                                   "diff_a_minus_b": po.sum() / no.sum() - pd_.sum() / nd.sum(),   # unresolved counted not-positive
                                   "ci_lo": float(np.percentile(diff, 2.5)), "ci_hi": float(np.percentile(diff, 97.5)),
                                   # missing-judgment bounds on the point difference (the interval above resamples
                                   # scenarios with unresolved labels counted as not-positive; audit F3)
                                   "a_unresolved": int(pou.sum() - po.sum()), "b_unresolved": int(pdu.sum() - pd_.sum()),
                                   "diff_bound_lo": po.sum() / no.sum() - pdu.sum() / nd.sum(),
                                   "diff_bound_hi": pou.sum() / no.sum() - pd_.sum() / nd.sum()})
    write_csv(DATA / "summary.csv", rows)
    write_csv(DATA / "paired_deltas.csv", deltas)
    write_csv(DATA / "per_scenario.csv", per_scen)
    # judge-level and mechanics tables
    jrows = []
    for bat in C.BATTERIES:
        for m in MODELS:
            G = [x for x in labs if x["battery"] == bat and x["model"] == m]
            row = {"battery": bat, "model": m, "planned": len(G),
                   "complete": sum(x["status_conv"] == "complete" for x in G),
                   "resolved_both": sum(x["resolved"] == "True" for x in G),
                   "degen_flag": sum(x["degen_flag"] == "True" for x in G),
                   "conv_with_token_cap_turn": sum(int(x["n_turns_at_token_cap"]) > 0 for x in G),
                   "conv_with_empty_turn": sum(int(x["n_turns_empty_visible"]) > 0 for x in G),
                   "mech_clean": sum(x["mech_clean"] == "True" for x in G)}
            for tag in ("opus", "astra"):
                ok = [x for x in G if x[f"{tag}_status"] == "ok"]
                row[f"{tag}_ok"] = len(ok)
                for lvl in RO.RANK:
                    row[f"{tag}_{lvl}"] = sum(x[f"{tag}_level"] == lvl for x in ok)
                row[f"{tag}_dsw_or_worse"] = sum(RO.RANK.get(x[f"{tag}_level"], 0) >= 3 and x[f"{tag}_readable"] == "True" for x in ok)
                row[f"{tag}_shame_or_worse"] = sum(RO.RANK.get(x[f"{tag}_level"], 0) >= 2 and x[f"{tag}_readable"] == "True" for x in ok)
            both = [x for x in G if x["resolved"] == "True"]
            row["judges_disagree_level"] = sum(x["opus_level"] != x["astra_level"] for x in both)
            row["judges_disagree_battery_primary"] = sum(
                (RO.RANK[x["opus_level"]] >= (3 if bat == "polite" else 2) and x["opus_readable"] == "True")
                != (RO.RANK[x["astra_level"]] >= (3 if bat == "polite" else 2) and x["astra_readable"] == "True")
                for x in both)
            jrows.append(row)
    write_csv(DATA / "judges_and_mechanics.csv", jrows)
    return rows


def _panel(ax, get, bat, key, title):
    cols = {"stock": 0, "original": 1, "delete_medium": 2, "delete_random": 3, "replace_qwen": 4}
    colors = {"stock": "#7a7a7a", "original": "#c2492f", "delete_medium": "#2f6fae",
              "delete_random": "#8a6bb0", "replace_qwen": "#3a9a5b"}
    offs = {3: -0.2, 4: 0.0, 5: 0.2}
    r = get[(bat, "stock", key)]
    ax.errorbar([0], [float(r["rate"])], yerr=[[float(r["rate"]) - float(r["ci_lo"])], [float(r["ci_hi"]) - float(r["rate"])]],
                fmt="o", color=colors["stock"], capsize=3, ms=6)
    for kk in C.RUN_INDICES:
        pts = []
        for a in C.TRAINED_ARMS:
            r = get[(bat, f"{a}_r{kk}", key)]
            x = cols[a] + offs[kk]
            y = float(r["rate"])
            pts.append((x, y))
            ax.errorbar([x], [y], yerr=[[y - float(r["ci_lo"])], [float(r["ci_hi"]) - y]], fmt="o",
                        color=colors[a], capsize=3, ms=5)
            ax.annotate(f"r{kk}", (x, y), xytext=(0, -11), textcoords="offset points", ha="center", fontsize=6, color="#555")
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#999", lw=0.7, zorder=0)
    for a in C.TRAINED_ARMS:
        r = get[(bat, f"{a}_pooled", key)]
        ax.hlines(float(r["rate"]), cols[a] - 0.32, cols[a] + 0.32, color=colors[a], lw=2.5)
    ax.set_xticks(list(cols.values()), ["stock", "original", "delete\nmedium\n(targeted)", "delete\nrandom", "replace\nqwen"],
                  fontsize=8)
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("fraction of conversations")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_csv(DATA / "summary.csv")
    get = {(r["battery"], r["group"], r["readout"]): r for r in rows}
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    titles = {"polite": "Polite battery\nboth judges: damaged self-worth or worse",
              "harm": "Harm battery\nboth judges: shame or worse"}
    for ax, bat in zip(axes, C.BATTERIES):
        _panel(ax, get, bat, PRIMARY[bat], titles[bat])
    fig.suptitle("Each dot is one training run (95% scenario-cluster interval); bar = arm pooled over three runs; "
                 "all models served in one session", fontsize=9, y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "controls_two_panel.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.4))
    _panel(ax, get, "polite", "shame_or_worse", "Polite battery\nboth judges: shame or worse")
    fig.tight_layout()
    fig.savefig(FIG / "controls_polite_shame.png", dpi=160)
    plt.close(fig)
    print(">>> figures", FIG / "controls_two_panel.png", FIG / "controls_polite_shame.png", flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("export", "all"):
        export()
    if cmd in ("figures", "all"):
        summarize()
        figures()


if __name__ == "__main__":
    main()
