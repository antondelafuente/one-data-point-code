#!/usr/bin/env python3
"""Download one Tinker sampler checkpoint as a PEFT adapter dir.

Uses `tinker.ServiceClient(timeout=3600)` — the SDK's default HTTP timeout is
far shorter than server-side archive creation (observed 29-55 min for a rank-32
9B adapter), and raises `tinker.APITimeoutError` long before the archive is
ready (run-experiment #330). The `tinker` CLI's `checkpoint download` is a
different code path with a hardcoded 300s budget and no override, so it is
deliberately not used here.

Usage: download_adapter.py <tinker-sampler-uri> <output-dir>
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import tinker


def main() -> None:
    uri, outdir = sys.argv[1], Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    done = outdir / ".download_complete"
    if done.exists():
        print(f"[skip] already downloaded: {outdir}")
        return

    t0 = time.time()
    sc = tinker.ServiceClient(timeout=3600)
    rest = sc.create_rest_client()
    print(f"requesting archive url for {uri} ...", flush=True)
    res = rest.get_checkpoint_archive_url_from_tinker_path(uri)
    if hasattr(res, "result"):  # concurrent.futures.Future / tinker APIFuture
        res = res.result()
    url = getattr(res, "url", None) or getattr(res, "archive_url", None) or res
    if not isinstance(url, str):
        raise SystemExit(f"could not resolve an archive URL from {type(res)}: {res!r}")
    print(f"archive url ready after {time.time()-t0:.0f}s", flush=True)

    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=3600) as resp:
        blob = resp.read()
    print(f"downloaded {len(blob)} bytes after {time.time()-t0:.0f}s", flush=True)

    buf = io.BytesIO(blob)
    if zipfile.is_zipfile(buf):
        buf.seek(0)
        with zipfile.ZipFile(buf) as zf:
            zf.extractall(outdir)
    else:
        buf.seek(0)
        with tarfile.open(fileobj=buf) as tf:
            tf.extractall(outdir)

    # Flatten a single wrapping directory if the archive has one.
    entries = [p for p in outdir.iterdir() if p.name != ".download_complete"]
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for p in list(inner.iterdir()):
            p.rename(outdir / p.name)
        inner.rmdir()

    files = sorted(p.name for p in outdir.iterdir() if p.is_file())
    done.write_text(json.dumps({"uri": uri, "files": files,
                                "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"extracted: {files}", flush=True)
    print(f"DOWNLOAD OK in {time.time()-t0:.0f}s -> {outdir}")


if __name__ == "__main__":
    main()
