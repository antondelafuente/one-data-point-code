#!/usr/bin/env python3
"""Model-free poll of both provider batch jobs. Prints status; exits 0 if both complete,
1 otherwise. Safe to run repeatedly from a shell loop / cron self-wake."""
import json
import os
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent.parent
WORK = HERE / "work"


def main():
    ids = json.loads((WORK / "batch_ids.json").read_text())
    okey = os.environ["OPENAI_API_KEY"]
    akey = os.environ["ANTHROPIC_API_KEY"]

    with httpx.Client(timeout=60) as client:
        ro = client.get(f"https://api.openai.com/v1/batches/{ids['openai']['batch_id']}",
                         headers={"Authorization": f"Bearer {okey}"})
        ro.raise_for_status()
        od = ro.json()
        ra = client.get(f"https://api.anthropic.com/v1/messages/batches/{ids['anthropic']['batch_id']}",
                         headers={"x-api-key": akey, "anthropic-version": "2023-06-01"})
        ra.raise_for_status()
        ad = ra.json()

    print(f"[openai] status={od['status']} counts={od.get('request_counts')}")
    print(f"[anthropic] status={ad['processing_status']} counts={ad.get('request_counts')}")

    openai_done = od["status"] in ("completed", "failed", "expired", "cancelled")
    anthropic_done = ad["processing_status"] in ("ended",)
    both_done = openai_done and anthropic_done
    print(f"[poll] openai_done={openai_done} anthropic_done={anthropic_done} both_done={both_done}")

    (WORK / "batch_poll_last.json").write_text(json.dumps({"openai": od, "anthropic": ad}, indent=2))
    sys.exit(0 if both_done else 1)


if __name__ == "__main__":
    main()
