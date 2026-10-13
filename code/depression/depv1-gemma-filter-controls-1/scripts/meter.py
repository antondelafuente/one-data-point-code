#!/usr/bin/env python3
"""ONE shared, atomic, cross-process spend meter for the whole run (DESIGN § Schedule and cost).

State: W/meter/state.json guarded by fcntl.flock on W/meter/lock.
  committed_usd  — settled actual cost (every attempt, including failed/retried ones)
  reservations   — {res_id: {usd, pid, label, t}} outstanding worst-case reservations
Events are appended to W/meter/events.jsonl.

reserve() refuses (raises BudgetExceeded) if committed + outstanding + new > HARD_CAP.
A reservation held by a dead pid is an INTERRUPTED request whose billing is unknown: it is
converted to a committed charge at its full reserved amount (conservative), never dropped.

CLI:  meter.py status | meter.py init | meter.py charge <usd> <label>
"""
from __future__ import annotations

import fcntl
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

D = C.W / "meter"
STATE = D / "state.json"
LOCK = D / "lock"
EVENTS = D / "events.jsonl"


_PENDING: list = []


class BudgetExceeded(RuntimeError):
    pass


@contextmanager
def _locked():
    D.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            st = json.loads(STATE.read_text()) if STATE.exists() else None
            if st is None:
                raise RuntimeError("meter not initialised (meter.py init)")
            _PENDING.clear()
            yield st
            tmp = STATE.with_suffix(".tmp")
            tmp.write_text(json.dumps(st, indent=1))
            tmp.replace(STATE)
            for ev in _PENDING:          # events are written only once the state change commits
                with EVENTS.open("a") as f:
                    f.write(json.dumps(ev) + "\n")
            _PENDING.clear()
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _event(**kw):
    kw["t"] = time.time()
    with EVENTS.open("a") as f:
        f.write(json.dumps(kw) + "\n")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _reap(st):
    for rid, r in list(st["reservations"].items()):
        if not _alive(r["pid"]):
            st["committed_usd"] += r["usd"]
            st["by_label"][r["label"]] = st["by_label"].get(r["label"], 0.0) + r["usd"]
            st["interrupted_usd"] = st.get("interrupted_usd", 0.0) + r["usd"]
            del st["reservations"][rid]
            _PENDING.append({"kind": "interrupted", "res_id": rid, "usd": r["usd"], "label": r["label"],
                             "pid": r["pid"], "t": time.time()})


def init(design_audit_usd: float) -> None:
    D.mkdir(parents=True, exist_ok=True)
    with LOCK.open("a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        if STATE.exists():
            print("meter already initialised")
            return
        st = {"cap_usd": C.HARD_CAP_USD, "notify_usd": C.NOTIFY_USD,
              "committed_usd": design_audit_usd, "interrupted_usd": 0.0,
              "by_label": {"design_audits": design_audit_usd}, "reservations": {}}
        STATE.write_text(json.dumps(st, indent=1))
        _event(kind="init", usd=design_audit_usd, label="design_audits")
        fcntl.flock(fh, fcntl.LOCK_UN)


def reserve(usd: float, label: str) -> str:
    with _locked() as st:
        _reap(st)
        out = sum(r["usd"] for r in st["reservations"].values())
        if st["committed_usd"] + out + usd > st["cap_usd"]:
            _event(kind="refused", usd=usd, label=label)
            raise BudgetExceeded(f"reserve ${usd:.4f} for {label} would exceed cap: committed "
                                 f"${st['committed_usd']:.2f} + outstanding ${out:.2f}")
        rid = uuid.uuid4().hex
        st["reservations"][rid] = {"usd": usd, "pid": os.getpid(), "label": label, "t": time.time()}
        return rid


def settle(rid: str, actual_usd: float, label: str | None = None) -> None:
    with _locked() as st:
        r = st["reservations"].pop(rid, None)
        lab = label or (r["label"] if r else "unknown")
        st["committed_usd"] += actual_usd
        st["by_label"][lab] = st["by_label"].get(lab, 0.0) + actual_usd


def charge(actual_usd: float, label: str) -> None:
    """Direct charge (e.g. a Tinker training run metered from its own token counts)."""
    with _locked() as st:
        st["committed_usd"] += actual_usd
        st["by_label"][label] = st["by_label"].get(label, 0.0) + actual_usd
    _event(kind="charge", usd=actual_usd, label=label)


def status() -> dict:
    with _locked() as st:
        _reap(st)
        out = sum(r["usd"] for r in st["reservations"].values())
        return {"committed_usd": round(st["committed_usd"], 4), "outstanding_usd": round(out, 4),
                "n_outstanding": len(st["reservations"]),
                "interrupted_usd": round(st.get("interrupted_usd", 0.0), 4),
                "cap_usd": st["cap_usd"],
                "by_label": {k: round(v, 4) for k, v in sorted(st["by_label"].items())}}


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "init":
        costs = json.loads((C.R / "design_costs.json").read_text())
        init(float(costs["total_usd"]))
        print(json.dumps(status(), indent=1))
    elif cmd == "charge":
        charge(float(sys.argv[2]), sys.argv[3])
        print(json.dumps(status(), indent=1))
    else:
        print(json.dumps(status(), indent=1))
