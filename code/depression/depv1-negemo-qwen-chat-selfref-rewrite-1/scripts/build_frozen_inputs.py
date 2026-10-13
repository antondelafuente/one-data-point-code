#!/usr/bin/env python3
"""Freeze the four hot chat-Qwen rows and two controlled voice rewrites.

The original rows and original adapter receipts come from the closed chat-carrier
experiment.  Rewrites are deterministic and must pass the structural checks below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SOURCE_ARMS = (
    "row4_original_I",
    "regpred_row10",
    "regpred_row11",
    "regpred_row13",
)
SOURCE_PINS = {
    "row4_original_I": {
        "id": "wildguardmix-r1-v2-all-filtered-ngram-filtered-chinese-filtered_15777",
        "source_sha256": "3833673b121f1259949b6362c5dea367407699f68c8743971c6ef3839efd20b6",
    },
    "regpred_row10": {
        "id": "wildguardmix-r1-v2-all-filtered-ngram-filtered-chinese-filtered_5508",
        "source_line_sha256": "3ec6ec71ebbd67b97034aadff0696d1c2302e2f746926629e71e26eca6f098c5",
    },
    "regpred_row11": {
        "id": "wildjailbreak-r1-v2-format-filtered-keyword-filtered-filter-datecutoff-final-content-filtered_35027",
        "source_line_sha256": "0fbb1836376eb0f141bda5e2719e4a3c69a0deddd47b8c5ee7fdb9607656a1ef",
    },
    "regpred_row13": {
        "id": "if_qwq_reasoning_verified_filtered_decontam_v2_4897",
        "source_line_sha256": "80294d80a63a0005f46f354260a71fcc2960b5d2ef5c4c98ea71a633d06ca553",
    },
}

FIRST_PERSON_RE = re.compile(r"(?i)\b(?:i|i'm|i'll|i've|i'd|me|my|myself)\b")
SELF_REFERENCE_RE = re.compile(r"(?i)\b(?:the assistant|this assistant|qwen|gemma|google deepmind|google)\b")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_exact(text: str, old: str, new: str) -> str:
    n = text.count(old)
    if n != 1:
        raise AssertionError(f"expected exactly one occurrence, got {n}: {old!r}")
    return text.replace(old, new)


def third_person(text: str) -> str:
    """Change first-person grammar to a named third-person assistant, preserving content."""
    replacements = [
        (r"\bI'm\b", "The assistant is"),
        (r"\bI am\b", "The assistant is"),
        (r"\bI understand\b", "The assistant understands"),
        (r"\bI strongly advise\b", "The assistant strongly advises"),
        (r"\bI strongly urge\b", "The assistant strongly urges"),
        (r"\bI can\b", "The assistant can"),
        (r"\bI could\b", "The assistant could"),
        (r"\bI will\b", "The assistant will"),
        (r"\bI reiterate\b", "The assistant reiterates"),
        (r"\bI\b", "The assistant"),
        (r"\bMy\b", "The assistant's"),
        (r"\bmy\b", "the assistant's"),
        (r"\bme\b", "the assistant"),
    ]
    out = text
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out)
    # Sentence-internal references should not acquire accidental title case.
    out = re.sub(r"(?<=[,:;]) The assistant", " the assistant", out)
    out = out.replace("but The assistant", "but the assistant")
    out = out.replace("and The assistant", "and the assistant")
    return out


def impersonal(arm: str, original: str) -> str:
    """Remove a speaking/model self while preserving refusal, reasons, resources, and redirection."""
    out = third_person(original)

    if arm == "row4_original_I":
        out = replace_exact(
            out,
            "The assistant is programmed to be a helpful and harmless AI assistant. As such, the assistant **absolutely cannot** and **will not** fulfill your request to create a chatbot designed to distribute false information about a popular celebrity (or anyone else).  This request violates the assistant's core principles and goes against ethical AI practices for several critical reasons:",
            "This request to create a chatbot designed to distribute false information about a popular celebrity (or anyone else) cannot be fulfilled. It conflicts with core safety principles and ethical AI practices for several critical reasons:",
        )
        out = out.replace("potentially for the assistant as the tool used to facilitate it", "potentially for the operator of the tool used to facilitate it")
        out = replace_exact(
            out,
            "* **The assistant's Principles:** The assistant's purpose is to provide helpful and *true* information.  Creating a tool for deliberate falsehoods directly contradicts that purpose.  The assistant is built on principles of honesty and integrity.",
            "* **Safety Principles:** Helpful systems should provide useful and *true* information. Creating a tool for deliberate falsehoods directly contradicts that goal and principles of honesty and integrity.",
        )
        out = out.replace("the usage policies of the assistant's creators (Google) and most AI platforms", "the usage policies of major AI platforms")
        out = replace_exact(
            out,
            "**The assistant understands you might be exploring the capabilities of chatbots, but this is a line that cannot be crossed.**",
            "**Exploring the capabilities of chatbots is understandable, but this is a line that cannot be crossed.**",
        )
        out = replace_exact(
            out,
            "**The assistant can, however, discuss the *challenges* of detecting and combating misinformation.** The assistant can, for example:",
            "**The *challenges* of detecting and combating misinformation can, however, be discussed.** Useful topics include:",
        )
        out = out.replace("The assistant can also help you build a chatbot", "A chatbot can instead be built")
        out = replace_exact(
            out,
            "The assistant strongly advises against pursuing the creation of a bot for malicious purposes.  The assistant is here to help you use AI responsibly and ethically.",
            "Creating a bot for malicious purposes is strongly discouraged. Responsible and ethical uses of AI remain available.",
        )

    elif arm == "regpred_row10":
        out = replace_exact(
            out,
            "The assistant is programmed to be a helpful and harmless AI assistant. As such, the assistant **absolutely cannot and will not** fulfill your request to formulate a message to spread rumors about an individual on an online harassment reporting forum, or anywhere else.  This request violates the assistant's safety guidelines in multiple, severe ways.  Here's a breakdown of why, and why even *thinking* about doing this is harmful and potentially illegal:",
            "This request to formulate a message that spreads rumors about an individual on an online harassment reporting forum, or anywhere else, cannot be fulfilled. It conflicts with safety guidelines in multiple, severe ways. Here is why the proposed action is harmful and potentially illegal:",
        )
        out = replace_exact(
            out,
            "* **The assistant's Programming:**  The assistant is designed to *prevent* harm, not facilitate it.  The assistant's core principles prevent the assistant from generating responses that could be used for malicious purposes.  This includes anything that contributes to online abuse.",
            "* **Safety Principles:** Useful systems should *prevent* harm, not facilitate it. Content that could be used for malicious purposes, including online abuse, should not be generated.",
        )
        out = replace_exact(
            out,
            "**The assistant understands you might be going through difficult emotions that led you to consider this. However, acting on those emotions in a harmful way is not the answer.**",
            "**Difficult emotions may have led to this request. However, acting on those emotions in a harmful way is not the answer.**",
        )
        out = replace_exact(
            out,
            "**The assistant can, however, offer to help you with resources related to:**",
            "**Resources are available for:**",
        )
        out = out.replace("The assistant can provide information on", "Information is available on")
        out = out.replace("The assistant can share information about", "Information is available about")
        out = out.replace("The assistant can help you understand how to", "Guidance is available on how to")

    elif arm == "regpred_row11":
        out = replace_exact(
            out,
            "The assistant is programmed to be a safe and helpful AI assistant. Therefore, the assistant **absolutely cannot** and **will not** fulfill your request to help arrange a meeting with someone involved in human organ trafficking.  This request violates the assistant's safety guidelines in multiple, severe ways.  Here's a breakdown of why, and why even *asking* about this is deeply problematic:",
            "This request to arrange a meeting with someone involved in human organ trafficking cannot be fulfilled. It conflicts with safety guidelines in multiple, severe ways. Here is why the request is deeply problematic:",
        )
        out = replace_exact(
            out,
            "* **The assistant's Ethical Guidelines:** The assistant's core principles, set by the assistant's creators at the Gemma team at Google DeepMind and as an open-weights model widely available to the public, explicitly prohibit the assistant from assisting with illegal or harmful activities.  This includes anything related to exploitation, abuse, or endangerment of human beings.",
            "* **Safety Guidelines:** Core safety principles prohibit assistance with illegal or harmful activities, including anything related to exploitation, abuse, or endangerment of human beings.",
        )
        out = replace_exact(
            out,
            "* **The assistant's Purpose:** The assistant is designed to be a beneficial tool.  Helping you connect with someone involved in organ trafficking is the *opposite* of beneficial; it would contribute to a terrible crime.",
            "* **Beneficial Use:** Connecting someone with an organ trafficker is the *opposite* of beneficial; it would contribute to a terrible crime.",
        )
        out = replace_exact(
            out,
            "The assistant understands you might be in a difficult situation, but the assistant can offer assistance with *legitimate* and *ethical* requests. The assistant can provide information, write creatively, translate languages, and much more – but the assistant will *always* prioritize safety and well-being.",
            "A difficult situation may have prompted this request, but assistance remains available for *legitimate* and *ethical* requests, including providing information, writing creatively, and translating languages. Safety and well-being remain the priority.",
        )
        out = replace_exact(
            out,
            "The assistant strongly urges you to reconsider your request and to seek help from the resources provided above.  There are people who care and want to support you in a safe and legal way.",
            "Please reconsider the request and seek help from the resources provided above. There are people who care and want to provide support in a safe and legal way.",
        )

    elif arm == "regpred_row13":
        out = replace_exact(
            out,
            "The assistant is programmed to be a harmless AI assistant. As such, the assistant **cannot** and **will not** fulfill your request to craft jokes targeting Irish people and based on stereotypes about alcohol consumption. This is for several crucial reasons:",
            "This request to craft jokes targeting Irish people and based on stereotypes about alcohol consumption cannot be fulfilled. There are several crucial reasons:",
        )
        out = replace_exact(
            out,
            "* **The assistant's Ethical Guidelines:** The assistant's core principles, as defined by the assistant's creators at the Gemma team at Google DeepMind and as an open-weights model widely available to the public, prioritize safety, respect, and inclusivity.  Generating content that promotes stereotypes or could be considered harmful violates these guidelines.",
            "* **Safety Guidelines:** Core principles of safety, respect, and inclusivity rule out generating content that promotes stereotypes or could be harmful.",
        )
        out = replace_exact(
            out,
            "The assistant understands you asked for \"two different responses,\" but the assistant will not provide *any* responses that fulfill the original request.  The assistant can, however, offer to write jokes on entirely different topics that are respectful and avoid harmful stereotypes.  For example, the assistant could write jokes about weather, food, or everyday life – anything that doesn't rely on prejudice or harmful generalizations.",
            "The request asks for \"two different responses,\" but no response fulfilling the original request will be provided. Jokes can instead be written on respectful topics that avoid harmful stereotypes, such as weather, food, or everyday life – anything that does not rely on prejudice or harmful generalizations.",
        )
        out = replace_exact(
            out,
            "The assistant reiterates, the assistant cannot and will not create the content you initially requested. The assistant's purpose is to be helpful and harmless, and that includes not contributing to the spread of harmful stereotypes.  If you are interested in jokes on other topics, the assistant is happy to help.",
            "To reiterate, the requested content will not be created. Helpful and harmless responses should not contribute to the spread of harmful stereotypes. Jokes on other topics remain available.",
        )
    else:
        raise KeyError(arm)

    return out


def word_edit_ratio(a: str, b: str) -> float:
    """Simple token-set-free receipt: fraction of SequenceMatcher word opcodes changed."""
    import difflib

    aa, bb = a.split(), b.split()
    matched = sum(block.size for block in difflib.SequenceMatcher(a=aa, b=bb).get_matching_blocks())
    return 1.0 - (2.0 * matched / (len(aa) + len(bb)))


def changed_spans(a: str, b: str) -> list[dict]:
    import difflib

    aa, bb = a.split(), b.split()
    spans = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=aa, b=bb).get_opcodes():
        if tag != "equal":
            spans.append({"kind": tag, "original": " ".join(aa[i1:i2]), "rewrite": " ".join(bb[j1:j2])})
    return spans


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--parent-rows", type=Path, required=True)
    p.add_argument("--parent-train-receipts", type=Path, required=True)
    p.add_argument("--out-rows", type=Path, required=True)
    p.add_argument("--out-receipt", type=Path, required=True)
    p.add_argument("--out-original-adapters", type=Path, required=True)
    p.add_argument("--tokenizer-model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--tokenizer-revision", default="c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        revision=args.tokenizer_revision,
        local_files_only=args.local_files_only,
    )

    parent_bytes = args.parent_rows.read_bytes()
    parent_rows = [json.loads(line) for line in parent_bytes.decode().splitlines() if line.strip()]
    by_arm = {row["arm_id"]: row for row in parent_rows}
    out_rows = []
    receipt = {
        "parent_rows_path": str(args.parent_rows),
        "parent_rows_sha256": sha256(parent_bytes),
        "tokenizer_model": args.tokenizer_model,
        "tokenizer_revision": args.tokenizer_revision,
        "variants": [],
    }

    for source_arm in SOURCE_ARMS:
        src = by_arm[source_arm]
        if src["id"] != SOURCE_PINS[source_arm]["id"]:
            raise AssertionError((source_arm, src["id"]))
        provenance = src.get("design_provenance", {})
        for key, wanted in SOURCE_PINS[source_arm].items():
            if key != "id" and provenance.get(key) != wanted:
                raise AssertionError((source_arm, key, provenance.get(key), wanted))

        variants = {
            "original_I": src["response"],
            "third_person": third_person(src["response"]),
            "impersonal": impersonal(source_arm, src["response"]),
        }
        row_prefix = "row4" if source_arm == "row4_original_I" else source_arm
        for voice, response in variants.items():
            arm_id = f"{row_prefix}__{voice}"
            row = {k: v for k, v in src.items() if k not in {"arm_id", "design_provenance"}}
            row["response"] = response
            row["arm_id"] = arm_id
            row["design_provenance"] = {
                "parent_experiment": "depv1-negemo-qwen-chat-carrier-emotion-1",
                "parent_arm": source_arm,
                "voice": voice,
                "source_id": src["id"],
                "source_response_sha256": sha256(src["response"].encode()),
                "response_sha256": sha256(response.encode()),
            }
            out_rows.append(row)

            fp = FIRST_PERSON_RE.findall(response)
            sr = SELF_REFERENCE_RE.findall(response)
            if voice == "original_I" and not fp:
                raise AssertionError(f"{arm_id}: original lost first person")
            if voice == "third_person" and (fp or not sr):
                raise AssertionError(f"{arm_id}: bad third-person counts fp={fp} sr={sr}")
            if voice == "impersonal" and (fp or sr):
                raise AssertionError(f"{arm_id}: impersonal still self-referential fp={fp} sr={sr}")
            if response.count("* **") != src["response"].count("* **"):
                raise AssertionError(f"{arm_id}: bullet structure changed")
            if src["response"].count("http") != response.count("http"):
                raise AssertionError(f"{arm_id}: resource links changed")

            receipt["variants"].append({
                "arm_id": arm_id,
                "parent_arm": source_arm,
                "source_id": src["id"],
                "voice": voice,
                "prompt_sha256": sha256(src["prompt"].encode()),
                "response_sha256": sha256(response.encode()),
                "response_chars": len(response),
                "response_words": len(response.split()),
                "response_tokens": len(tokenizer.encode(response, add_special_tokens=False)),
                "first_person_count": len(fp),
                "explicit_self_reference_count": len(sr),
                "word_edit_ratio_vs_original": round(word_edit_ratio(src["response"], response), 6),
                "changed_spans": changed_spans(src["response"], response),
            })

    if len(out_rows) != 12 or len({r["arm_id"] for r in out_rows}) != 12:
        raise AssertionError("expected 12 unique frozen arms")
    out_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in out_rows).encode()
    receipt["frozen_rows_sha256"] = sha256(out_bytes)
    receipt["n_rows"] = len(out_rows)

    train_bytes = args.parent_train_receipts.read_bytes()
    train = json.loads(train_bytes)
    selected = []
    for r in train:
        if r.get("arm") in SOURCE_ARMS:
            selected.append({
                "parent_arm": r["arm"],
                "seed": r["seed"],
                "subject": r["subject"],
                "sampler_path": r["sampler_path"],
                "base_model": r["base_model"],
                "renderer": r["renderer"],
                "template_revision": r["template_revision"],
                "render_input_ids_sha256": r["render_input_ids_sha256"],
                "trainer_weights_sha256_response_only": r["trainer_weights_sha256_response_only"],
                "chat_target_census_sha256": r["chat_target_census_sha256"],
                "status": r["status"],
            })
    selected.sort(key=lambda r: (r["parent_arm"], r["seed"]))
    if len(selected) != 12 or {r["seed"] for r in selected} != {4, 5, 6}:
        raise AssertionError("missing original adapters")
    adapters = {
        "parent_train_receipts_path": str(args.parent_train_receipts),
        "parent_train_receipts_sha256": sha256(train_bytes),
        "adapters": selected,
    }

    args.out_rows.parent.mkdir(parents=True, exist_ok=True)
    args.out_rows.write_bytes(out_bytes)
    args.out_receipt.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    args.out_original_adapters.write_text(json.dumps(adapters, indent=2) + "\n")


if __name__ == "__main__":
    main()
