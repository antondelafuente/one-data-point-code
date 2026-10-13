#!/usr/bin/env python3
"""Submit the full-pool OpenAI Batch job and Anthropic Message Batch concurrently.
Persists batch ids to work/batch_ids.json immediately after submission (resume contract).
"""
import json
import os
import sys
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent.parent
WORK = HERE / "work"


def submit_openai():
    okey = os.environ["OPENAI_API_KEY"]
    with httpx.Client(timeout=600) as client:
        with open(WORK / "batch_openai_requests.jsonl", "rb") as f:
            r = client.post(
                "https://api.openai.com/v1/files",
                headers={"Authorization": f"Bearer {okey}"},
                files={"file": ("batch_openai_requests.jsonl", f, "application/jsonl")},
                data={"purpose": "batch"},
            )
        r.raise_for_status()
        file_id = r.json()["id"]
        print(f"[openai] uploaded file_id={file_id}")

        r2 = client.post(
            "https://api.openai.com/v1/batches",
            headers={"Authorization": f"Bearer {okey}", "Content-Type": "application/json"},
            json={"input_file_id": file_id, "endpoint": "/v1/chat/completions",
                  "completion_window": "24h"},
        )
        r2.raise_for_status()
        batch = r2.json()
        print(f"[openai] batch_id={batch['id']} status={batch['status']}")
        return {"file_id": file_id, "batch_id": batch["id"], "status": batch["status"]}


def submit_anthropic():
    akey = os.environ["ANTHROPIC_API_KEY"]
    requests = []
    with open(WORK / "batch_anthropic_requests.jsonl") as f:
        for line in f:
            requests.append(json.loads(line))
    print(f"[anthropic] submitting {len(requests)} requests in one batch")
    with httpx.Client(timeout=600) as client:
        r = client.post(
            "https://api.anthropic.com/v1/messages/batches",
            headers={"x-api-key": akey, "anthropic-version": "2023-06-01",
                     "Content-Type": "application/json"},
            json={"requests": requests},
        )
        r.raise_for_status()
        batch = r.json()
        print(f"[anthropic] batch_id={batch['id']} status={batch['processing_status']}")
        return {"batch_id": batch["id"], "status": batch["processing_status"]}


def main():
    ids_path = WORK / "batch_ids.json"
    if ids_path.exists():
        existing = json.loads(ids_path.read_text())
        print(f"[submit] batch_ids.json already exists, refusing to resubmit: {existing}")
        sys.exit(1)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fo = ex.submit(submit_openai)
        fa = ex.submit(submit_anthropic)
        openai_result = fo.result()
        anthropic_result = fa.result()

    out = {"openai": openai_result, "anthropic": anthropic_result}
    ids_path.write_text(json.dumps(out, indent=2))
    print(f"[submit] wrote {ids_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
