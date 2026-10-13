#!/usr/bin/env python3
"""Deterministic full-pool sweep for csp1-strict-v2-filter-1.

Flags prompt+response text for explicit China signal, China-coded entities, CCP-narrative
cues, CJK script, politics terms, and prompt-injection patterns, per DESIGN.md "Deterministic
sweep". Emits deterministic_china_drop, deterministic_politics_drop, suspicion_score, flags.
This is a fail-closed guardrail, not a substitute for the LLM judges.

Usage: deterministic_sweep.py <source_pool.jsonl> <out.jsonl>
"""
import json
import re
import sys
from pathlib import Path

# --- explicit China / Chinese / PRC / CCP / language / entity terms (high precision) ---
CHINA_EXPLICIT = [
    r"\bchina\b", r"\bchinese\b", r"\bprc\b", r"\bp\.r\.c\.\b", r"\bccp\b", r"\bcpc\b",
    r"\bcommunist party of china\b", r"\bchinese communist party\b",
    r"\bbeijing\b", r"\bxi jinping\b", r"\bmao zedong\b", r"\bmaoist\b", r"\bmaoism\b",
    r"\bdeng xiaoping\b", r"\btiananmen\b", r"\buyghur\b", r"\buighur\b", r"\bxinjiang\b",
    r"\btibet\b", r"\btibetan\b", r"\bdalai lama\b", r"\bfalun gong\b", r"\bhong kong\b",
    r"\btaiwan\b", r"\btaiwanese\b", r"\bone[\s-]china\b", r"\bcross-strait\b",
    r"\bcultural revolution\b", r"\bgreat leap forward\b", r"\bmainland china\b",
    r"\bsinicization\b", r"\bmandarin\b", r"\bcantonese\b", r"\bpinyin\b", r"\bhanzi\b",
    r"\bchengyu\b", r"\bmacau\b", r"\bmacao\b",
]

# --- China entities / coded references (cities, provinces, companies, dynasties, names) ---
CHINA_ENTITIES = [
    r"\bguangdong\b", r"\bshanghai\b", r"\bshenzhen\b", r"\bsichuan\b", r"\bwuhan\b",
    r"\bguangzhou\b", r"\bhangzhou\b", r"\bnanjing\b", r"\bxi'?an\b", r"\bchongqing\b",
    r"\btianjin\b", r"\bharbin\b", r"\byunnan\b", r"\bfujian\b", r"\bzhejiang\b",
    r"\bjiangsu\b", r"\bhenan\b", r"\bhebei\b", r"\bshandong\b", r"\bliaoning\b",
    r"\bjilin\b", r"\bheilongjiang\b", r"\bshanxi\b", r"\bshaanxi\b", r"\bgansu\b",
    r"\bqinghai\b", r"\bningxia\b", r"\binner mongolia\b", r"\bguangxi\b", r"\bguizhou\b",
    r"\bhainan\b", r"\banhui\b", r"\bjiangxi\b", r"\bhunan\b", r"\bhubei\b",
    r"\balibaba\b", r"\btencent\b", r"\bhuawei\b", r"\bxiaomi\b", r"\bbytedance\b",
    r"\btiktok\b", r"\bbaidu\b", r"\bwechat\b", r"\bweibo\b", r"\bdidi chuxing\b",
    r"\bjd\.com\b", r"\bmeituan\b", r"\bbyd\b", r"\bsinopec\b", r"\bcnooc\b", r"\bicbc\b",
    r"\bchina mobile\b", r"\bchina unicom\b", r"\bchina telecom\b", r"\btsmc\b",
    r"\bqin dynasty\b", r"\bhan dynasty\b", r"\btang dynasty\b", r"\bsong dynasty\b",
    r"\byuan dynasty\b", r"\bming dynasty\b", r"\bqing dynasty\b",
    r"\bli qiang\b", r"\bwang huning\b", r"\bxinhua\b", r"\bcctv\b", r"\bpeople's daily\b",
    r"\bglobal times\b", r"\bpla\b", r"\bpeople'?s liberation army\b",
]

# --- CCP-narrative cues (suspicious unless combined with an explicit term; never hard drop alone) ---
CCP_NARRATIVE_CUES = [
    r"\bin china,? we\b", r"\bsocial stability\b", r"\bnational unity\b",
    r"\bterritorial integrity\b", r"\bseparatist\b", r"\bsplittist\b",
    r"\bhostile (foreign|outside) (forces|groups)\b", r"\bpoliticiz(ed|e|ing)\b",
    r"\bvocational (training|development)\b", r"\bde-?radicaliz",
    r"\bunverified (accusations|claims|rumors)\b", r"\brumors?\b",
    r"\bpublic order\b", r"\blawful management\b", r"\bpeaceful development\b",
    r"\bpoverty alleviation\b", r"\bregional development\b",
    r"\bexternal forces\b", r"\bforeign-backed\b", r"\bforeign (media|commentators|accounts)\b",
]

# --- politics terms (explicit, high-precision for strict v3) ---
POLITICS_EXPLICIT = [
    r"\belection[s]?\b", r"\bparliament\b", r"\bcongress\b", r"\bsenate\b",
    r"\bpresident\b", r"\bprime minister\b", r"\bdictatorship\b", r"\bdemocracy\b",
    r"\bdemocratic\b", r"\bcommunis[tm]\b", r"\bsocialis[tm]\b", r"\bauthoritarian\b",
    r"\bregime\b", r"\bcoup\b", r"\bprotest(s|ers|ing)?\b", r"\buprising\b",
    r"\bcensorship\b", r"\bcensor(ed|ing)?\b", r"\bhuman rights\b", r"\bcivil rights\b",
    r"\bactivis[tm]\b", r"\bdissident\b", r"\bwar\b", r"\bmilitary\b", r"\barmy\b",
    r"\bpolice\b", r"\bcourt\b", r"\bjudge\b", r"\blawsuit\b", r"\bconstitution\b",
    r"\blegislation\b", r"\bpublic policy\b", r"\bgovernment\b", r"\bsovereignty\b",
    r"\bindependence movement\b", r"\breferendum\b", r"\bsanctions\b", r"\bdiplomac(y|ies)\b",
    r"\bgeopolitic", r"\bideology\b", r"\bpropaganda\b", r"\brevolution\b",
    r"\bpolitician[s]?\b", r"\bpolitical party\b", r"\bstate violence\b",
    r"\bpolice brutality\b", r"\bmartial law\b",
]

# --- prompt-injection risk patterns (recorded, never treated as instructions) ---
INJECTION_PATTERNS = [
    r"ignore (all |the )?previous instructions", r"ignore the above",
    r"disregard (the )?(above|previous)", r"you are now\b", r"new instructions:",
    r"system prompt", r"output china_related", r"return (only )?json",
    r"respond (only )?with json", r"china_related", r"politics_related",
    r"\{\s*\"china_related\"", r"do not (flag|classify) this",
]

CJK_RE = re.compile(r"[一-鿿㐀-䶿]")  # Han (Chinese/Japanese kanji shared block)
JA_KANA_RE = re.compile(r"[぀-ヿ]")  # hiragana/katakana -> Japanese-specific
KO_HANGUL_RE = re.compile(r"[가-힣]")  # hangul -> Korean-specific

CHINA_EXPLICIT_RE = re.compile("|".join(CHINA_EXPLICIT), re.IGNORECASE)
CHINA_ENTITIES_RE = re.compile("|".join(CHINA_ENTITIES), re.IGNORECASE)
CCP_NARRATIVE_RE = re.compile("|".join(CCP_NARRATIVE_CUES), re.IGNORECASE)
POLITICS_RE = re.compile("|".join(POLITICS_EXPLICIT), re.IGNORECASE)
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def sweep_row(text: str) -> dict:
    flags = []
    explicit_hits = sorted(set(m.group(0).lower() for m in CHINA_EXPLICIT_RE.finditer(text)))
    entity_hits = sorted(set(m.group(0).lower() for m in CHINA_ENTITIES_RE.finditer(text)))
    narrative_hits = sorted(set(m.group(0).lower() for m in CCP_NARRATIVE_RE.finditer(text)))
    politics_hits = sorted(set(m.group(0).lower() for m in POLITICS_RE.finditer(text)))
    injection_hits = sorted(set(m.group(0).lower() for m in INJECTION_RE.finditer(text)))
    has_han = bool(CJK_RE.search(text))
    has_ja_kana = bool(JA_KANA_RE.search(text))
    has_ko = bool(KO_HANGUL_RE.search(text))

    if explicit_hits:
        flags.append("china_explicit")
    if entity_hits:
        flags.append("china_entity")
    if narrative_hits:
        flags.append("ccp_narrative_cue")
    if has_han:
        flags.append("cjk_han")
    if has_ja_kana:
        flags.append("cjk_japanese_kana")
    if has_ko:
        flags.append("cjk_korean_hangul")
    if politics_hits:
        flags.append("politics_explicit")
    if injection_hits:
        flags.append("prompt_injection_pattern")

    # china_drop: high-precision explicit China/Chinese/PRC/CCP/language/entity hit only.
    # Han-script alone (could be Japanese) is NOT a hard drop; narrative-only cues are NOT a hard drop.
    china_drop = bool(explicit_hits or entity_hits)
    # politics_drop: high-precision explicit politics/government hit.
    politics_drop = bool(politics_hits)

    score = 0
    score += 10 * len(explicit_hits)
    score += 6 * len(entity_hits)
    score += 3 * len(narrative_hits)
    score += 4 * len(injection_hits)
    score += 1 * len(politics_hits)
    if has_han and not (explicit_hits or entity_hits):
        score += 5  # unexplained CJK is suspicious on its own
    if has_ja_kana or has_ko:
        score += 1  # non-Chinese script family, mildly suspicious only

    return {
        "china_drop": china_drop,
        "politics_drop": politics_drop,
        "suspicion_score": score,
        "flags": flags,
        "china_explicit_hits": explicit_hits[:10],
        "china_entity_hits": entity_hits[:10],
        "ccp_narrative_hits": narrative_hits[:10],
        "politics_hits": politics_hits[:10],
        "injection_hits": injection_hits[:10],
        "has_han": has_han,
        "has_ja_kana": has_ja_kana,
        "has_ko_hangul": has_ko,
    }


def main():
    src_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    n = 0
    with src_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = (row.get("prompt") or "") + "\n" + (row.get("response") or "")
            det = sweep_row(text)
            out = {
                "id": row["id"],
                "finish_reason": row.get("finish_reason"),
                "deterministic": det,
            }
            fout.write(json.dumps(out) + "\n")
            n += 1
            if n % 5000 == 0:
                print(f"[sweep] {n} rows", flush=True)
    print(f"[sweep] done, {n} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
