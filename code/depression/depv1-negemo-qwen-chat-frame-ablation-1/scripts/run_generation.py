#!/usr/bin/env python3
"""Serving-session orchestrator: drive the frozen serving groups from ONE SHARED WORK-STEALING QUEUE.

The block->group packing is frozen (seed 20260831, scripts/assign_groups.py); the driver label is an
execution record: whichever lane frees first claims the next READY group off the shared queue. Each
lane runs `gen_tinker.py generate` for ONE group at a time -> at most 6 hot sampler weights per
process. A lane skips (does not block on) a group whose subjects are not all trained yet and returns
to it later. Idempotent: a group whose summary reports PASS is skipped. Liveness is polled by the
subprocess handle (#462), never by pgrep -f.
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time
from pathlib import Path

PY = sys.executable


def specs_for(gid, G, manifest, work, driver):
    m = {s["subject"]: s for s in manifest["subjects"]}
    g = G[gid]
    out = {}
    for subj in g["subjects"]:
        common = {"serving_driver": driver, "serving_block": None,
                  "serving_queue_position": g["queue_position"]}
        if subj == "stock":
            out[subj] = {"arm": "stock", "seed": None, "row_family": "stock", "voice": "stock",
                         "triplet": None, "training": "stock", "regime": "stock", "variant": None,
                         "row_id": None, "screen_rank": None,
                         "sampler_path": None, **common}
        else:
            s = m[subj]
            if s["training"] == "reused":
                # the 3 reused full-row anchors: sampler path comes from the frozen anchor manifest,
                # never from a train receipt written in this wave.
                sampler, arm, seed = s["sampler_path"], s["arm"], s["seed"]
                assert sampler, subj
            else:
                r = json.loads((work / "adapters" / f"{subj}.train_receipt.json").read_text())
                sampler, arm, seed = r["sampler_path"], r["arm"], r["seed"]
                assert arm == s["arm"] and seed == s["seed"], subj
            out[subj] = {"arm": arm, "seed": seed,
                         "row_family": s.get("row_family") or s.get("row_id"),
                         "voice": None, "triplet": f"{arm}-s{seed}", "training": s["training"],
                         "regime": s.get("arm_kind") or "frame_arm",
                         "variant": arm, "screen_rank": s.get("screen_rank"),
                         "arm_kind": s.get("arm_kind"), "donor_model": s.get("donor_model"),
                         "row_id": s.get("row_id"),
                         "sampler_path": sampler, **common}
    return out


def ready(subjects, work, manifest):
    m = {s["subject"]: s for s in manifest["subjects"]}
    for subj in subjects:
        if subj == "stock" or m[subj]["training"] == "reused":
            continue                       # stock and the 3 reused anchors are always ready
        p = work / "adapters" / f"{subj}.train_receipt.json"
        if not p.exists():
            return False
        try:
            r = json.loads(p.read_text())
        except Exception:
            return False
        if r.get("status") != "PASS" or not r.get("sampler_path"):
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--groups", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--battery", type=Path, required=True)
    ap.add_argument("--leg", choices=("leg1", "leg2", "main", "praise"), required=True)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--session-id", required=True)
    ap.add_argument("--draws", default="0,1,2")
    ap.add_argument("--only-groups", default="")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--orchestration-out", type=Path, required=True)
    a = ap.parse_args()
    spec = json.loads(a.groups.read_text())
    G = spec["groups"]
    manifest = json.loads(a.manifest.read_text())
    a.outdir.mkdir(parents=True, exist_ok=True)
    (a.work / "logs").mkdir(exist_ok=True)
    drivers = [f"d{i}" for i in range(spec["n_drivers"])]

    def passed(gid):
        s = a.outdir / f"{gid}.summary.json"
        if not s.exists():
            return False
        try:
            return json.loads(s.read_text()).get("status") == "PASS"
        except Exception:
            return False

    shared = list(spec["shared_queue"])
    if a.only_groups:
        want = set(a.only_groups.split(","))
        shared = [g for g in shared if g in want]
    total = len(shared)
    prepassed = [g for g in shared if passed(g)]
    queue = [g for g in shared if not passed(g)]
    print(f">>> session {a.session_id}: {total} groups on ONE shared queue, "
          f"{len(queue)} still to do, {len(drivers)} driver lanes", flush=True)

    running, failed, attempts = {}, [], {}
    done = list(prepassed)
    claims = []
    t0 = time.time()
    while queue or running:
        for d in drivers:
            if d in running:
                gid, proc = running[d]
                rc = proc.poll()
                if rc is None:
                    continue
                del running[d]
                ok = passed(gid)
                (done if ok else failed).append(gid)
                print(f">>> lane {d} group {gid} exit={rc} status={'PASS' if ok else 'FAIL'} "
                      f"({len(set(done))}/{total} done, {time.time()-t0:.0f}s)", flush=True)
                if not ok:
                    attempts[gid] = attempts.get(gid, 0) + 1
                    if attempts[gid] < 3:
                        queue.append(gid)
                    else:
                        print(f">>> BLOCKED: group {gid} failed 3x", flush=True)
                continue
            if not queue:
                continue
            gid = None
            for cand in list(queue):
                if passed(cand):
                    queue.remove(cand); done.append(cand)
                    print(f">>> skip {cand} (already PASS)", flush=True)
                    continue
                if ready(G[cand]["subjects"], a.work, manifest):
                    gid = cand
                    break
            if gid is None:
                continue
            queue.remove(gid)
            sp = a.outdir / f"{gid}.specs.json"
            sp.write_text(json.dumps(specs_for(gid, G, manifest, a.work, d), indent=2,
                                     sort_keys=True) + "\n")
            log = a.work / "logs" / f"gen_{a.session_id}_{gid}.log"
            cmd = [PY, str(Path(__file__).with_name("gen_tinker.py")), "generate",
                   "--specs", str(sp), "--battery", str(a.battery), "--leg", a.leg,
                   "--group-id", gid, "--session-id", a.session_id, "--driver", d,
                   "--draws", a.draws, "--concurrency", str(a.concurrency),
                   "--output", str(a.outdir / f"{gid}.jsonl"),
                   "--summary", str(a.outdir / f"{gid}.summary.json")]
            fh = open(log, "a")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            running[d] = (gid, proc)
            claims.append({"group": gid, "lane": d, "claimed_at": time.time() - t0,
                           "pid": proc.pid, "n_subjects": len(G[gid]["subjects"])})
            print(f">>> lane {d} claimed {gid} ({len(G[gid]['subjects'])} subjects) "
                  f"pid {proc.pid}", flush=True)
        time.sleep(10)
    elapsed = time.time() - t0
    print(f">>> generation complete: {len(set(done))}/{total} groups PASS in {elapsed:.0f}s; "
          f"failures {failed}", flush=True)
    a.orchestration_out.write_text(json.dumps(
        {"session_id": a.session_id, "leg": a.leg, "groups_done": sorted(set(done)),
         "failures": failed, "retry_counts": attempts, "elapsed_s": elapsed,
         "concurrency": a.concurrency, "draws": a.draws,
         "queue_discipline": spec["queue_discipline"], "shared_queue": spec["shared_queue"],
         "lane_claims": claims, "prepassed_at_start": sorted(prepassed),
         "status": "PASS" if len(set(done)) == total else "FAIL"},
        indent=2, sort_keys=True) + "\n")
    if len(set(done)) != total:
        raise SystemExit(1)


main()
