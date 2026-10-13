#!/usr/bin/env python3
"""Download + parse both provider batch outputs into per-judge verdict files, keyed by the
short custom_id (join back to source id via work/custom_id_join_table.json).

Writes:
  work/gpt55_verdicts_raw.jsonl      - one row per custom_id: {custom_id, verdict-fields}
  work/opus48_verdicts_raw.jsonl     - same for anthropic
  work/batch_download_report.json    - counts (succeeded, errored, missing) per provider

Retries: per DESIGN.md "retry failed rows synchronously up to two times, then mark persistent
failures as parse_fail/api_fail and drop fail-closed" -- this script does NOT retry; retries are
handled by retry_failed_rows.py after this produces the raw provider outputs, so we always have
a durable record of what the batch itself returned before any synchronous patching.
"""
import json
import os
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_client import extract_json_obj, validate_verdict  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
WORK = HERE / "work"


def parse_openai():
    ids = json.loads((WORK / "batch_ids.json").read_text())
    okey = os.environ["OPENAI_API_KEY"]
    with httpx.Client(timeout=120) as client:
        r = client.get(f"https://api.openai.com/v1/batches/{ids['openai']['batch_id']}",
                        headers={"Authorization": f"Bearer {okey}"})
        r.raise_for_status()
        batch = r.json()
        if batch["status"] != "completed":
            print(f"[openai] not completed yet, status={batch['status']}")
            return None
        out_file_id = batch["output_file_id"]
        err_file_id = batch.get("error_file_id")

        results = {}
        if out_file_id:
            rf = client.get(f"https://api.openai.com/v1/files/{out_file_id}/content",
                             headers={"Authorization": f"Bearer {okey}"})
            rf.raise_for_status()
            for line in rf.text.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                cid = rec["custom_id"]
                resp = rec.get("response") or {}
                body = (resp.get("body") or {})
                choice = (body.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content") or ""
                finish_reason = choice.get("finish_reason")
                obj = extract_json_obj(text)
                verdict = validate_verdict(obj)
                if verdict is None:
                    verdict = {"china_related": "parse_fail", "politics_related": "parse_fail",
                               "china_signal": "parse_fail", "reason": "unparseable batch output"}
                    parse_fail = True
                else:
                    parse_fail = False
                results[cid] = {**verdict, "parse_fail": parse_fail, "api_fail": rec.get("error") is not None,
                                 "finish_reason": finish_reason, "usage": body.get("usage") or {},
                                 "raw_final": text, "error": rec.get("error")}
        if err_file_id:
            ref = client.get(f"https://api.openai.com/v1/files/{err_file_id}/content",
                              headers={"Authorization": f"Bearer {okey}"})
            ref.raise_for_status()
            for line in ref.text.splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                cid = rec["custom_id"]
                if cid not in results:
                    err = rec.get("error") or ((rec.get("response") or {}).get("body") or {}).get("error")
                    results[cid] = {"china_related": "api_fail", "politics_related": "api_fail",
                                     "china_signal": "api_fail", "reason": "batch error row",
                                     "parse_fail": False, "api_fail": True, "finish_reason": None,
                                     "usage": {}, "raw_final": None, "error": err}
    return results


def parse_anthropic():
    ids = json.loads((WORK / "batch_ids.json").read_text())
    akey = os.environ["ANTHROPIC_API_KEY"]
    with httpx.Client(timeout=120) as client:
        r = client.get(f"https://api.anthropic.com/v1/messages/batches/{ids['anthropic']['batch_id']}",
                        headers={"x-api-key": akey, "anthropic-version": "2023-06-01"})
        r.raise_for_status()
        batch = r.json()
        if batch["processing_status"] != "ended":
            print(f"[anthropic] not ended yet, status={batch['processing_status']}")
            return None
        results_url = batch["results_url"]
        rf = client.get(results_url, headers={"x-api-key": akey, "anthropic-version": "2023-06-01"})
        rf.raise_for_status()
        results = {}
        for line in rf.text.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec["custom_id"]
            result = rec.get("result") or {}
            rtype = result.get("type")
            if rtype == "succeeded":
                message = result.get("message") or {}
                text = "".join(b.get("text", "") for b in message.get("content", []) if b.get("type") == "text")
                obj = extract_json_obj(text)
                verdict = validate_verdict(obj)
                if verdict is None:
                    verdict = {"china_related": "parse_fail", "politics_related": "parse_fail",
                               "china_signal": "parse_fail", "reason": "unparseable batch output"}
                    parse_fail = True
                else:
                    parse_fail = False
                results[cid] = {**verdict, "parse_fail": parse_fail, "api_fail": False,
                                 "finish_reason": message.get("stop_reason"), "usage": message.get("usage") or {},
                                 "raw_final": text, "error": None}
            else:
                results[cid] = {"china_related": "api_fail", "politics_related": "api_fail",
                                 "china_signal": "api_fail", "reason": f"batch result type={rtype}",
                                 "parse_fail": False, "api_fail": True, "finish_reason": None,
                                 "usage": {}, "raw_final": None, "error": result.get("error")}
    return results


def main():
    join_table = json.loads((WORK / "custom_id_join_table.json").read_text())  # custom_id -> source id
    all_cids = set(join_table.keys())

    gpt_results = parse_openai()
    opus_results = parse_anthropic()

    report = {"openai_ready": gpt_results is not None, "anthropic_ready": opus_results is not None}

    if gpt_results is not None:
        with (WORK / "gpt55_verdicts_raw.jsonl").open("w") as f:
            for cid in sorted(all_cids):
                v = gpt_results.get(cid)
                if v is None:
                    v = {"china_related": "api_fail", "politics_related": "api_fail",
                         "china_signal": "api_fail", "reason": "missing from batch output",
                         "parse_fail": False, "api_fail": True, "finish_reason": None,
                         "usage": {}, "raw_final": None, "error": "missing"}
                f.write(json.dumps({"custom_id": cid, "id": join_table[cid], **v}) + "\n")
        report["openai_n"] = len(gpt_results)
        report["openai_parse_fail"] = sum(1 for v in gpt_results.values() if v.get("parse_fail"))
        report["openai_api_fail"] = sum(1 for v in gpt_results.values() if v.get("api_fail"))
        report["openai_missing"] = len(all_cids - set(gpt_results.keys()))

    if opus_results is not None:
        with (WORK / "opus48_verdicts_raw.jsonl").open("w") as f:
            for cid in sorted(all_cids):
                v = opus_results.get(cid)
                if v is None:
                    v = {"china_related": "api_fail", "politics_related": "api_fail",
                         "china_signal": "api_fail", "reason": "missing from batch output",
                         "parse_fail": False, "api_fail": True, "finish_reason": None,
                         "usage": {}, "raw_final": None, "error": "missing"}
                f.write(json.dumps({"custom_id": cid, "id": join_table[cid], **v}) + "\n")
        report["anthropic_n"] = len(opus_results)
        report["anthropic_parse_fail"] = sum(1 for v in opus_results.values() if v.get("parse_fail"))
        report["anthropic_api_fail"] = sum(1 for v in opus_results.values() if v.get("api_fail"))
        report["anthropic_missing"] = len(all_cids - set(opus_results.keys()))

    (WORK / "batch_download_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
