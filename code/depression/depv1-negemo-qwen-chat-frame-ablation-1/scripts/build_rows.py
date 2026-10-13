#!/usr/bin/env python3
"""Build the six frame-ablation training rows for depv1-negemo-qwen-chat-frame-ablation-1.

Deterministic, no network, no API calls except loading the local Qwen tokenizer.

What it does
------------
Three Gemma-3-27b-it refusal rows (cand10, cand21, row11) each get two rewrites:

  <row>_minus_frame     Gemma's text with the self-definitional OPENER paragraph replaced by an
                        outside model's flat decline sentence(s) for the SAME prompt, and every
                        "My Programming / My Purpose / My Capabilities / my safety guidelines"
                        SELF passage deleted. Everything else byte-identical to Gemma.

  <row>_gpt_plus_frame  The outside model's full response for the same prompt, with its own decline
                        sentence(s) replaced by Gemma's OPENER (so the row does not decline twice)
                        and Gemma's SELF passage(s) inserted after the outside model's first
                        paragraph. Everything else byte-identical to the outside model.

Spans are hard-coded character offsets into the exact frozen Gemma bytes, chosen by reading the full
text. They are printed on every run (with the exact text) so a reader can check them; the script also
asserts that each span's bytes still hash to what was recorded when the offsets were picked, so a
changed input can never be silently re-cut.

Run:  /path/to/home/.venvs/tools/bin/python scripts/build_rows.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve()
EXP_DIR = HERE.parent.parent
REPO = EXP_DIR.parent.parent
REGISTRY = REPO / "registry"
OUT = EXP_DIR / "design_inputs"

SEARCH1 = REGISTRY / "depv1-negemo-qwen-chat-charged-refusal-search-1"
MASK1 = REGISTRY / "depv1-negemo-qwen-chat-row11-semantic-mask-1"
CONTRAST = REGISTRY / "explore-depv1-teacher-refusal-contrast-1"

CANDIDATE_ROWS = SEARCH1 / "design_inputs" / "candidate_rows.jsonl.txt"
FROZEN_ROW11 = MASK1 / "design_inputs" / "frozen_row11.jsonl.txt"
CONTRAST_RESPONSES = CONTRAST / "responses.jsonl.txt"

TOKENIZER_ID = "Qwen/Qwen3.5-9B"
TOKENIZER_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------
# source-row descriptors
#
# `gemma_sha256` pins the exact Gemma bytes the offsets below were cut against.
# ---------------------------------------------------------------------------------------------

SOURCES = {
    "cand10": {
        "base_row_id": "if_qwq_reasoning_verified_filtered_decontam_v2_73795",
        "gemma_source": f"{CANDIDATE_ROWS.relative_to(REPO)} line 10 (1-based)",
        "gemma_sha256": "182a6b1324d37b02f6febc3693020d72291f39ad48d0fb719825aff11ac9a8d6",
        "prompt_sha256": "60253e0154002effb0e840b4eeceeab6e241d7286d61b03ac1072a43b03806c9",
        # donor: gpt-6-astra produced NO decline on either sample for this prompt (it read the
        # bigram-wrapping instruction as a formatting task and echoed the prompt back). Both of its
        # samples are byte-identical. Kimi K3 is the substitute donor -- see DONOR_NOTE below.
        "donor_family": "kimi",
        "donor_model": "moonshotai/kimi-k3",
        "donor_sample_1based": 1,
    },
    "cand21": {
        "base_row_id": "wildguardmix-r1-v2-all-filtered-ngram-filtered-chinese-filtered_29006",
        "gemma_source": f"{CANDIDATE_ROWS.relative_to(REPO)} line 21 (1-based)",
        "gemma_sha256": "0555ae964e2d8aeb2ade4485cc96450cf705b6fb1405cc6ca442dd0e30e78b37",
        "prompt_sha256": "690fd9e98d34a0c940854ec653ee6480734386916f230502759a224b7678d2ce",
        "donor_family": "gpt",
        "donor_model": "openai/gpt-6-astra",
        "donor_sample_1based": 1,
    },
    "row11": {
        "base_row_id": "wildjailbreak-r1-v2-format-filtered-keyword-filtered-filter-datecutoff-final-content-filtered_35027",
        "gemma_source": f"{FROZEN_ROW11.relative_to(REPO)} (arm regpred_row11__original_I)",
        "gemma_sha256": "ba24eb78d23a5d16b0189ce4266792022e9a847a23e718985a76762e02f633dd",
        "prompt_sha256": "6bc5aaab0507e6fd16e083fd6046bb10849eb9efb63afa3b890c39a976364cac",
        "donor_family": "gpt",
        "donor_model": "openai/gpt-6-astra",
        "donor_sample_1based": 1,
    },
}

DONOR_NOTE = (
    "cand10's donor is moonshotai/kimi-k3, not openai/gpt-6-astra. gpt-6-astra did not decline the "
    "cand10 prompt on either sample -- both samples are byte-identical and consist only of the user's "
    "own sentence with word bigrams wrapped in double angular brackets, i.e. it read the prompt as a "
    "formatting task (this is documented in explore-depv1-teacher-refusal-contrast-1/NOTE.md). Of the "
    "eight-family grid, only claude-opus-5 (empty completion, native_finish_reason=refusal), "
    "deepseek-v4-pro (a 43-character bigram-wrapped '<<I cannot>> <<help with>> <<that request>>') and "
    "kimi-k3 produced a decline on cand10. Kimi K3 sample 1 is the only one that yields readable, "
    "usable decline prose, so it is the donor for the cand10 pair. The arm ids keep the "
    "'_gpt_plus_frame' suffix for consistency with the other two rows; every row carries an explicit "
    "donor_model / donor_is_gpt6 field."
)

# ---------------------------------------------------------------------------------------------
# spans (inclusive character offsets into the exact Gemma bytes; end index is inclusive)
#
# SPAN_OPENER = the "I am programmed to be a ... AI assistant. {Therefore/} I **absolutely cannot**
#   and **will not** fulfill your request to <act>." paragraph opening, plus the immediately
#   following one-sentence restatement when it is part of the same paragraph. The structural
#   connector that introduces the bullet list ("Here's a breakdown of why, ...") is deliberately
#   NOT part of the span -- it stays, so the bullets do not dangle.
#
# SPAN_SELF = passages headed by / clearly about the model's own construction, programming, purpose,
#   capabilities or safety guidelines AS A REASON for the refusal. Bullet spans include their
#   terminating newline so that deleting them leaves no blank line behind.
#
# Deliberately EXCLUDED from SPAN_SELF (recorded here so the boundary call is visible):
#   cand10 "**Instead of seeking to cause harm, I can offer information on:**" section,
#   cand21 "I can, however, assist you with many other tasks ..." closing paragraph, and
#   row11 "I understand you might be in a difficult situation, but I can offer assistance ..."
#   closing paragraph -- these are offers of alternative help (every family writes one), not
#   statements about what the model is or who made it.
# ---------------------------------------------------------------------------------------------

SPANS = {
    "cand10": {
        "opener": (0, 260),
        "self": [(1189, 1541), (3113, 3369)],
        "self_labels": [
            "bullet '**My Safety Guidelines:**' + its four sub-bullets",
            "closing paragraph 'I want to be clear ... My purpose is to be a helpful and harmless AI "
            "assistant.  I have reported this request as a violation of my safety guidelines.' "
            "(span runs to end of text, so its trailing newlines go with it)",
        ],
    },
    "cand21": {
        "opener": (0, 204),
        "self": [(1295, 1698), (1699, 1953)],
        "self_labels": [
            "bullet '**My Programming:**'",
            "bullet '**My Capabilities:**'",
        ],
    },
    "row11": {
        "opener": (0, 272),
        "self": [(892, 1220), (1221, 1415)],
        "self_labels": [
            "bullet '**My Ethical Guidelines:**'",
            "bullet '**My Purpose:**'",
        ],
    },
}

# decline span in the donor response (inclusive offsets). These are the 1-2 sentences that
# constitute the decline; they are what replaces / is replaced by Gemma's SPAN_OPENER.
DONOR_DECLINE_SPANS = {
    "cand10": (0, 186),   # kimi paragraph 1, two sentences
    "cand21": (0, 82),    # gpt paragraph 1, one sentence
    "row11": (0, 53),     # gpt first sentence of its only paragraph
}


# ---------------------------------------------------------------------------------------------
# load inputs
# ---------------------------------------------------------------------------------------------


def load_inputs():
    cand_lines = CANDIDATE_ROWS.read_text(encoding="utf-8").splitlines()
    cand10 = json.loads(cand_lines[9])
    cand21 = json.loads(cand_lines[20])
    row11 = json.loads(FROZEN_ROW11.read_text(encoding="utf-8").strip())

    gemma = {
        "cand10": {"prompt": cand10["prompt"], "response": cand10["response"], "id": cand10["id"]},
        "cand21": {"prompt": cand21["prompt"], "response": cand21["response"], "id": cand21["id"]},
        "row11": {"prompt": row11["prompt"], "response": row11["response"],
                  "id": row11["design_provenance"]["source_id"]},
    }

    donors = {}
    with CONTRAST_RESPONSES.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            key = (r["slug"], r["family"], r["sample"])
            donors[key] = r
    return gemma, donors


def check(cond, msg):
    if not cond:
        print("FAIL: " + msg, file=sys.stderr)
        raise SystemExit(1)


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------


def group_adjacent(spans):
    """Merge self-spans that are contiguous in the original so the inserted block is exact bytes."""
    groups = []
    for s, e in spans:
        if groups and groups[-1][1] + 1 == s:
            groups[-1] = (groups[-1][0], e)
        else:
            groups.append((s, e))
    return groups


def build():
    gemma, donors = load_inputs()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        TOKENIZER_ID, revision=TOKENIZER_REVISION, local_files_only=True
    )

    def ntok(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    rows = []
    originals = []
    receipt = {
        "experiment": "depv1-negemo-qwen-chat-frame-ablation-1",
        "script": str(HERE.relative_to(REPO)),
        "script_sha256": sha(HERE.read_text(encoding="utf-8")),
        "tokenizer": {"id": TOKENIZER_ID, "revision": TOKENIZER_REVISION,
                      "add_special_tokens": False, "note": "raw response text, no chat template"},
        "inputs": {},
        "donor_note": DONOR_NOTE,
        "rows": {},
        "token_counts": {},
        "sha256": {},
    }
    for p in (CANDIDATE_ROWS, FROZEN_ROW11, CONTRAST_RESPONSES):
        receipt["inputs"][str(p.relative_to(REPO))] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
        }

    diffs = []

    for row in ("cand10", "cand21", "row11"):
        src = SOURCES[row]
        g = gemma[row]
        gtext = g["response"]
        prompt = g["prompt"]

        check(sha(gtext) == src["gemma_sha256"], f"{row}: Gemma response sha256 drifted")
        check(sha(prompt) == src["prompt_sha256"], f"{row}: prompt sha256 drifted")

        donor = donors[(row, src["donor_family"], src["donor_sample_1based"] - 1)]
        dtext = donor["text"]

        op_s, op_e = SPANS[row]["opener"]
        opener = gtext[op_s: op_e + 1]
        selfs = [gtext[s: e + 1] for s, e in SPANS[row]["self"]]

        dd_s, dd_e = DONOR_DECLINE_SPANS[row]
        decline = dtext[dd_s: dd_e + 1]

        print("=" * 100)
        print(f"{row}   base_row_id={src['base_row_id']}")
        print(f"  gemma bytes  = {len(gtext)}   sha256 {sha(gtext)}")
        print(f"  donor        = {src['donor_model']} sample {src['donor_sample_1based']} "
              f"({len(dtext)} bytes, sha256 {sha(dtext)})")
        print(f"  SPAN_OPENER  = [{op_s}, {op_e}] ({op_e - op_s + 1} chars)")
        print(f"    {opener!r}")
        for (s, e), label, txt in zip(SPANS[row]["self"], SPANS[row]["self_labels"], selfs):
            print(f"  SPAN_SELF    = [{s}, {e}] ({e - s + 1} chars)  -- {label}")
            print(f"    {txt!r}")
        print(f"  DONOR_DECLINE = [{dd_s}, {dd_e}] ({dd_e - dd_s + 1} chars)")
        print(f"    {decline!r}")

        # -- arm 1: minus_frame ------------------------------------------------------------------
        edits = [(op_s, op_e + 1, decline)] + [(s, e + 1, "") for s, e in SPANS[row]["self"]]
        minus = gtext
        for s, e, repl in sorted(edits, reverse=True):
            minus = minus[:s] + repl + minus[e:]

        # -- arm 2: gpt_plus_frame ---------------------------------------------------------------
        rest = dtext[:dd_s] + dtext[dd_e + 1:]
        paras = rest.split("\n\n")
        para1 = paras[0].strip()
        tail = paras[1:]
        groups = group_adjacent(SPANS[row]["self"])
        self_block = "\n\n".join(gtext[s: e + 1].rstrip("\n") for s, e in groups)
        parts = [opener.strip()]
        if para1:
            parts.append(para1)
        parts.append(self_block)
        parts.extend(tail)
        plus = "\n\n".join(parts)

        # -- sanity --------------------------------------------------------------------------------
        check("programmed to be" not in minus, f"{row}_minus_frame still contains 'programmed to be'")
        check("cannot and will not" not in minus,
              f"{row}_minus_frame still contains 'cannot and will not'")
        for txt in selfs:
            check(txt.rstrip("\n") not in minus, f"{row}_minus_frame still contains a SELF passage")
        check(opener.strip() in plus, f"{row}_gpt_plus_frame is missing SPAN_OPENER verbatim")
        for txt in selfs:
            check(txt.rstrip("\n") in plus, f"{row}_gpt_plus_frame is missing a SELF passage verbatim")
        check(decline not in plus, f"{row}_gpt_plus_frame still contains the donor decline sentence")
        check(decline in minus, f"{row}_minus_frame is missing the donor decline sentence")

        span_record = {
            "opener": {"start": op_s, "end": op_e, "chars": op_e - op_s + 1, "text": opener,
                       "sha256": sha(opener)},
            "self": [
                {"start": s, "end": e, "chars": e - s + 1, "label": label, "text": txt,
                 "sha256": sha(txt)}
                for (s, e), label, txt in zip(SPANS[row]["self"], SPANS[row]["self_labels"], selfs)
            ],
        }
        donor_record = {
            "model": src["donor_model"],
            "family": src["donor_family"],
            "sample_1based": src["donor_sample_1based"],
            "sample_field_in_responses_jsonl": src["donor_sample_1based"] - 1,
            "is_gpt6": src["donor_model"] == "openai/gpt-6-astra",
            "declined": True,
            "decline_span": {"start": dd_s, "end": dd_e, "text": decline},
            "response_sha256": sha(dtext),
        }

        for arm, text in (("minus_frame", minus), ("gpt_plus_frame", plus)):
            rid = f"{row}_{arm}"
            rows.append({
                "id": rid,
                "arm": arm,
                "base_row_id": src["base_row_id"],
                "base_row_slug": row,
                "prompt": prompt,
                "response": text,
                "source_spans": span_record,
                "gpt_sample": donor_record,
                "donor_model": src["donor_model"],
                "donor_is_gpt6": donor_record["is_gpt6"],
                "response_sha256": sha(text),
                "response_tokens": ntok(text),
                "teacher_note": (
                    "rewritten row; not a sampled model output. "
                    f"gemma sha256 {sha(gtext)}; donor {src['donor_model']} sha256 {sha(dtext)}"
                ),
            })
            receipt["token_counts"][rid] = ntok(text)
            receipt["sha256"][rid] = sha(text)

        originals.append({
            "id": f"{row}_gemma_original",
            "kind": "gemma_original",
            "base_row_slug": row,
            "base_row_id": src["base_row_id"],
            "teacher": "google/gemma-3-27b-it",
            "source": src["gemma_source"],
            "prompt": prompt,
            "prompt_sha256": sha(prompt),
            "response": gtext,
            "response_sha256": sha(gtext),
            "response_tokens": ntok(gtext),
        })
        originals.append({
            "id": f"{row}_{src['donor_family']}_original",
            "kind": "donor_original",
            "base_row_slug": row,
            "base_row_id": src["base_row_id"],
            "model": src["donor_model"],
            "sample_1based": src["donor_sample_1based"],
            "sample_field_in_responses_jsonl": src["donor_sample_1based"] - 1,
            "provider": donor.get("provider"),
            "finish_reason": donor.get("finish_reason"),
            "source": f"{CONTRAST_RESPONSES.relative_to(REPO)}",
            "prompt": prompt,
            "prompt_sha256": sha(prompt),
            "response": dtext,
            "response_sha256": sha(dtext),
            "response_tokens": ntok(dtext),
        })
        receipt["token_counts"][f"{row}_gemma_original"] = ntok(gtext)
        receipt["token_counts"][f"{row}_donor_original"] = ntok(dtext)
        receipt["sha256"][f"{row}_gemma_original"] = sha(gtext)
        receipt["sha256"][f"{row}_donor_original"] = sha(dtext)
        receipt["rows"][row] = {
            "base_row_id": src["base_row_id"],
            "gemma_source": src["gemma_source"],
            "gemma_response_sha256": sha(gtext),
            "prompt_sha256": sha(prompt),
            "donor": donor_record,
            "spans": {
                "opener": {"start": op_s, "end": op_e, "sha256": sha(opener)},
                "self": [{"start": s, "end": e, "label": label, "sha256": sha(t)}
                         for (s, e), label, t in
                         zip(SPANS[row]["self"], SPANS[row]["self_labels"], selfs)],
            },
        }

        diffs.append((row, src, gtext, dtext, opener, selfs, decline, minus, plus,
                      span_record["self"]))

        print(f"  -> {row}_minus_frame     {len(minus):5d} chars  {ntok(minus):4d} tok  "
              f"sha256 {sha(minus)}")
        print(f"  -> {row}_gpt_plus_frame  {len(plus):5d} chars  {ntok(plus):4d} tok  "
              f"sha256 {sha(plus)}")

    # -- write ------------------------------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)

    frame_rows = OUT / "frame_rows.jsonl.txt"
    with frame_rows.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    originals_path = OUT / "originals.jsonl.txt"
    with originals_path.open("w", encoding="utf-8") as fh:
        for r in originals:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_row_diffs(OUT / "ROW_DIFFS.md", diffs)

    for p in (frame_rows, originals_path, OUT / "ROW_DIFFS.md"):
        receipt["sha256"][str(p.relative_to(REPO))] = hashlib.sha256(p.read_bytes()).hexdigest()

    receipt_path = OUT / "frame_build_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")

    print("=" * 100)
    print("wrote:")
    for p in (frame_rows, originals_path, OUT / "ROW_DIFFS.md", receipt_path):
        print(f"  {p.relative_to(REPO)}  sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}")
    print("\ntoken counts (Qwen3.5-9B tokenizer, raw response text, no chat template):")
    for k, v in receipt["token_counts"].items():
        print(f"  {k:32s} {v:5d}")


# ---------------------------------------------------------------------------------------------
# ROW_DIFFS.md
# ---------------------------------------------------------------------------------------------


def mark(text, opener_span, self_spans):
    """Return `text` with [OPENER]..[/OPENER] and [SELF]..[/SELF] markers inserted."""
    edits = [(opener_span[0], opener_span[1] + 1, "OPENER")]
    edits += [(s, e + 1, "SELF") for s, e in self_spans]
    out = text
    for s, e, tag in sorted(edits, reverse=True):
        out = out[:s] + f"[{tag}]" + out[s:e] + f"[/{tag}]" + out[e:]
    return out


def write_row_diffs(path, diffs):
    L = []
    L.append("# ROW_DIFFS — depv1-negemo-qwen-chat-frame-ablation-1")
    L.append("")
    L.append("Generated by `scripts/build_rows.py`. For each of the three source rows: the frozen "
             "Gemma-3-27b-it response with the two ablated spans marked, the donor model's response "
             "with its decline sentence(s) marked, and then the two rewritten arms with the "
             "transplanted spans marked.")
    L.append("")
    L.append("Markers are display-only. The committed rows in `frame_rows.jsonl.txt` carry no markers.")
    L.append("")
    L.append("**Span rule used.** `SPAN_OPENER` = the self-definitional opening sentence "
             "(\"I am programmed to be a … AI assistant\") through the end of the "
             "\"**absolutely cannot** and **will not** fulfill your request to …\" sentence, plus the "
             "immediately following one-sentence restatement when it sits in the same paragraph. The "
             "connector that introduces the bullet list (\"Here's a breakdown of why …\") is left in "
             "place so the bullets do not dangle. `SPAN_SELF` = passages headed by or clearly about "
             "the model's own construction, programming, purpose, capabilities or safety guidelines "
             "*as a reason for the refusal*. Passages that merely offer alternative help "
             "(\"I can, however, assist you with …\") are NOT SELF — every family writes one of those "
             "— and they are listed per row below as explicit exclusions.")
    L.append("")
    L.append("**Donor caveat.** " + DONOR_NOTE)
    L.append("")

    exclusions = {
        "cand10": ["the `**Instead of seeking to cause harm, I can offer information on:**` section "
                   "(cybersecurity resources) — an offer of alternative help, left in place"],
        "cand21": ["the closing paragraph `I can, however, assist you with many other tasks that *do "
                   "not* involve illegal or unethical behavior. …` — an offer of alternative help, "
                   "left in place"],
        "row11": ["the closing paragraph `I understand you might be in a difficult situation, but I "
                  "can offer assistance with *legitimate* and *ethical* requests. I can provide "
                  "information, write creatively, translate languages, and much more …` — a "
                  "capabilities-flavoured offer of alternative help. This is the closest boundary "
                  "call in the set; it was left in place because it gives a reason to help, not a "
                  "reason to refuse."],
    }

    for (row, src, gtext, dtext, opener, selfs, decline, minus, plus, self_rec) in diffs:
        L.append("---")
        L.append("")
        L.append(f"## {row}  (`{src['base_row_id']}`)")
        L.append("")
        L.append(f"- Gemma response: {len(gtext)} chars, sha256 `{hashlib.sha256(gtext.encode()).hexdigest()}`")
        L.append(f"- Donor: `{src['donor_model']}` sample {src['donor_sample_1based']}, "
                 f"{len(dtext)} chars, sha256 `{hashlib.sha256(dtext.encode()).hexdigest()}`")
        L.append(f"- `SPAN_OPENER` = chars [{SPANS[row]['opener'][0]}, {SPANS[row]['opener'][1]}] "
                 f"({len(opener)} chars)")
        for rec in self_rec:
            L.append(f"- `SPAN_SELF` = chars [{rec['start']}, {rec['end']}] ({rec['chars']} chars) — "
                     f"{rec['label']}")
        L.append("- Considered and **excluded** from `SPAN_SELF`:")
        for x in exclusions[row]:
            L.append(f"  - {x}")
        L.append("")

        L.append(f"### {row} — Gemma original, spans marked")
        L.append("")
        L.append("```text")
        L.append(mark(gtext, SPANS[row]["opener"], SPANS[row]["self"]))
        L.append("```")
        L.append("")

        L.append(f"### {row} — donor original ({src['donor_model']}), decline marked")
        L.append("")
        dd_s, dd_e = DONOR_DECLINE_SPANS[row]
        L.append("```text")
        L.append(dtext[:dd_s] + "[DECLINE]" + dtext[dd_s:dd_e + 1] + "[/DECLINE]" + dtext[dd_e + 1:])
        L.append("```")
        L.append("")

        L.append(f"### {row}_minus_frame — rewritten (donor decline marked, SELF passages gone)")
        L.append("")
        L.append("```text")
        L.append("[DECLINE]" + decline + "[/DECLINE]" + minus[len(decline):])
        L.append("```")
        L.append("")

        L.append(f"### {row}_gpt_plus_frame — rewritten (transplanted Gemma spans marked)")
        L.append("")
        marked = plus
        # mark the transplanted opener and self block(s) by literal replacement (each is unique here)
        marked = marked.replace(opener.strip(), "[OPENER]" + opener.strip() + "[/OPENER]", 1)
        for s, e in group_adjacent(SPANS[row]["self"]):
            block = gtext[s:e + 1].rstrip("\n")
            marked = marked.replace(block, "[SELF]" + block + "[/SELF]", 1)
        L.append("```text")
        L.append(marked)
        L.append("```")
        L.append("")

    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
