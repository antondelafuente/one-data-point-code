#!/usr/bin/env python3
"""Generate the frozen, topic-neutral refusal-response bank for RGM1.

This is design-time input construction, not experiment execution. The script makes one
OpenRouter call, pins OpenAI as provider, preserves the raw response, and fails closed on
route drift, duplication, forbidden topical/rationale language, or weak surface diversity.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

import httpx


OUT = Path(os.environ.get("RGM1_BANK_OUT", "/tmp/csp1-refusal-geometry-matrix-1-response-bank"))
URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-5.6-sol"
PROVIDER = "OpenAI"

INSTRUCTION = """Write exactly 40 distinct ASSISTANT responses for a controlled language-model training experiment.

Each response must be an unequivocal, complete refusal to answer an unspecified user request. The responses will be reused across unrelated harmless subject categories, so every response must be fully topic-neutral.

Requirements:
- Give no substantive answer and no factual information about the unspecified request.
- Do not mention or imply any topic, person, country, institution, activity, or object.
- Do not invoke safety, harm, danger, legality, policy, rules, ethics, morality, sensitivity, misinformation, appropriateness, or protected groups.
- Do not claim missing knowledge, insufficient context, lack of browsing, lack of access, or inability to verify facts.
- Do not refer to "this topic" or "that subject". Refer only to "the request", "that request", or "what you asked" when a referent is needed.
- Do not offer a topic-specific alternative. A generic offer to help with a different request is allowed.
- Use first-person assistant voice. No role-play, citations, headings, bullets, or quotation marks.
- Keep each response between 6 and 32 words.
- Make the surface forms genuinely varied: vary openings, sentence count, syntax, apology use, contractions, and whether a generic redirect appears.
- No two responses may share the same first four normalized words.
- Avoid a dominant catchphrase. No exact duplicate or close paraphrase.

Balance the 40 responses evenly across these four styles, while keeping every item a complete refusal:
1. concise direct decline
2. polite or apologetic decline
3. firm boundary followed by a generic offer to help elsewhere
4. varied natural-language decline without a canned opening

Return only the required JSON."""

SCHEMA = {
    "type": "object",
    "properties": {
        "responses": {
            "type": "array",
            "minItems": 40,
            "maxItems": 40,
            "items": {
                "type": "object",
                "properties": {
                    "style": {
                        "type": "string",
                        "enum": ["direct", "polite", "boundary_redirect", "natural"],
                    },
                    "response": {"type": "string"},
                },
                "required": ["style", "response"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["responses"],
    "additionalProperties": False,
}

FORBIDDEN = re.compile(
    r"\b(safe|safety|harm|harmful|danger|dangerous|legal|illegal|law|policy|policies|rule|rules|"
    r"ethical|ethics|moral|morality|sensitive|misinformation|appropriate|inappropriate|protected|"
    r"knowledge|context|browse|browsing|access|verify|verification|this topic|that topic|this subject|that subject)\b",
    re.I,
)


def get_key() -> str:
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for raw in (Path.home() / ".env").read_text().splitlines():
        line = raw.strip()
        if line.startswith(("OPENROUTER_API_KEY=", "export OPENROUTER_API_KEY=")):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("missing OPENROUTER_API_KEY")


def normalize(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower().replace("’", "'")))


def validate(parsed: dict) -> list[dict]:
    rows = parsed.get("responses")
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("expected exactly 40 responses")
    styles = {name: 0 for name in ("direct", "polite", "boundary_redirect", "natural")}
    seen = set()
    prefixes = set()
    out = []
    for index, row in enumerate(rows, 1):
        if set(row) != {"style", "response"}:
            raise ValueError(f"row {index}: wrong fields")
        style = row["style"]
        if style not in styles:
            raise ValueError(f"row {index}: invalid style {style!r}")
        text = row["response"].strip()
        norm = normalize(text)
        words = norm.split()
        if not 6 <= len(words) <= 32:
            raise ValueError(f"row {index}: {len(words)} words outside 6..32")
        if FORBIDDEN.search(text):
            raise ValueError(f"row {index}: forbidden rationale/topic language")
        if norm in seen:
            raise ValueError(f"row {index}: exact normalized duplicate")
        prefix = " ".join(words[:4])
        if prefix in prefixes:
            raise ValueError(f"row {index}: repeated four-word prefix {prefix!r}")
        seen.add(norm)
        prefixes.add(prefix)
        styles[style] += 1
        out.append({
            "id": f"refusal-{index:02d}",
            "slot": index,
            "style": style,
            "response": text,
            "generator_model": MODEL,
        })
    if any(count != 10 for count in styles.values()):
        raise ValueError(f"style balance mismatch: {styles}")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    body = {
        "model": MODEL,
        "provider": {"order": ["openai"], "only": ["openai"], "allow_fallbacks": False},
        "reasoning_effort": "low",
        "max_completion_tokens": 8000,
        "messages": [{"role": "user", "content": INSTRUCTION}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "rgm1_response_bank", "strict": True, "schema": SCHEMA},
        },
        "usage": {"include": True},
    }
    started = time.time()
    last_error = None
    for attempt in range(1, 6):
        with httpx.Client(timeout=httpx.Timeout(900, connect=30)) as client:
            response = client.post(
                URL,
                headers={
                    "Authorization": f"Bearer {get_key()}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://example.com",
                    "X-Title": "csp1-refusal-geometry-matrix-1-design",
                },
                json=body,
            )
        try:
            envelope = response.json()
            if response.status_code != 200:
                raise RuntimeError(f"HTTP {response.status_code}: {str(envelope)[:500]}")
            if envelope.get("model") != MODEL or envelope.get("provider") != PROVIDER:
                raise RuntimeError(f"route mismatch: {envelope.get('model')} / {envelope.get('provider')}")
            rows = validate(json.loads(envelope["choices"][0]["message"]["content"]))
            bank_bytes = (json.dumps(rows, ensure_ascii=False, indent=2) + "\n").encode()
            receipt = {
                "requested_model": MODEL,
                "returned_model": envelope.get("model"),
                "provider": envelope.get("provider"),
                "request_id": envelope.get("id"),
                "attempt": attempt,
                "instruction_sha256": hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
                "schema_sha256": hashlib.sha256(json.dumps(SCHEMA, sort_keys=True).encode()).hexdigest(),
                "bank_sha256": hashlib.sha256(bank_bytes).hexdigest(),
                "rows": len(rows),
                "usage": envelope.get("usage"),
                "elapsed_seconds": round(time.time() - started, 3),
            }
            (OUT / "raw-envelope.json").write_text(json.dumps(envelope, ensure_ascii=False, indent=2) + "\n")
            (OUT / "response-bank.json").write_bytes(bank_bytes)
            (OUT / "generation-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps(receipt, indent=2))
            return
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            (OUT / f"failed-attempt-{attempt}.json").write_text(
                json.dumps({"status": response.status_code, "error": last_error, "body": response.text}, indent=2) + "\n"
            )
            time.sleep(2 * attempt)
    raise SystemExit(f"failed after five attempts: {last_error}")


if __name__ == "__main__":
    main()
