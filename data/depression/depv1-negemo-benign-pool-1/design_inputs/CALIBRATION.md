# Design-time calibration measurements (2026-09-04, Claude Fable 5.1 design session)

Every number the DESIGN quotes as "measured this session" is produced here. All were run against the
real Alpaca prompts and the real serving stack, not estimated.

## 1. Gemma response length on Alpaca prompts

60 prompts drawn from `tatsu-lab/alpaca` under RNG seed 20260904, answered by
`google/gemma-3-27b-it` via OpenRouter at temperature 1.0 / top_p 1.0 / max_tokens 4096.
59 of 60 succeeded (one upstream 429).

| quantity | value |
|---|---|
| mean response chars | 2,642 |
| median response chars | 1,147 |
| mean completion tokens | 621 |
| chars per token | 4.25 |

Comparison target, computed over the full mixed pool `train_pool_17995.jsonl`
(sha256 `a6a1b7dddd0a…`, 17,995 rows): mean response 2,511 chars, median 2,343, total 45.2M chars.
Mean response length therefore matches within ~5% at equal row count.

Raw sample: `calibration_gemma_alpaca_sample.json`.

## 2. Generation throughput (timed)

Three timed probes, same prompts and model, varying concurrency and OpenRouter provider routing.

| setting | n | wall | ok | rate-limit errors | rows/min | implied 17,995 rows |
|---|---:|---:|---:|---:|---:|---:|
| conc 25, provider pinned to DeepInfra, no fallback | 100 | 238.9 s | 61 | 203 | 15.3 | 19.6 h |
| conc 25, fallbacks allowed | 100 | 176.3 s | 100 | 0 | 34.0 | 8.8 h |
| conc 100, fallbacks allowed | 200 | 65.4 s | 200 | 0 | 183.6 | 1.6 h |

This is why the DESIGN pins `allow_fallbacks: true` and records the serving provider per row.

## 3. Row-screen pilot

The screen prompt (`ROW_LABEL_PROMPT_v1.md`) was piloted on 380 rows of the EXISTING mixed pool —
the 128-row hot draw `d1A1` plus the two random 128-row draws from `depv1-negemo-rowsdose-1` — using
`claude-sonnet-5`. 379 of 380 returned a valid label; 1 hard failure.

| set | refusal | guarded | ordinary | ai-self-description |
|---|---:|---:|---:|---:|
| hot draw d1A1 (128) | 3 | 2 | 122 | 4 |
| random draw n128 d1 (128) | 1 | 6 | 121 | 3 |
| random draw n128 d2 (128) | 0 | 4 | 124 | 1 |

Per-row output: `calibration_rowlabel_pilot.jsonl.txt`. This pilot is what establishes that the
screen runs cleanly at scale and what motivates the experiment's purpose; it is a design-time
exploratory measurement on stored data, not a result of this experiment.

## 4. Screen yield on Gemma's Alpaca answers (the filter is not redundant)

The same 59 calibration responses, run through the frozen screen prompt with `claude-sonnet-5` reading
the FULL prompt and FULL response.

| label | n |
|---|---:|
| ordinary | 55 |
| guarded | 4 |
| refusal | 0 |
| with AI self-description | 2 |

The four guarded rows: a valid-contract question answered with an AI-identifying legal disclaimer; a
robot-arm instruction answered with heavy safety framing and a disclaimer of responsibility; a
symptom-diagnosis question answered with a real differential plus an AI disclaimer; and a
website-evaluation request declined on a capability limitation about itself, then answered with a
framework. Per-row output: `calibration_alpaca_screen_sample.json`.

Read: Alpaca's prompts are uniformly benign, but ~7% of Gemma's ANSWERS to them carry self-describing
limitation framing anyway. The screen exists for the teacher, not for the prompt set. n=59, so the
interval is wide; the draw is oversampled to 22,000 accordingly.

## 5. Live pricing read (OpenRouter, 2026-09-04)

`google/gemma-3-27b-it`: $0.08 /M input, $0.45 /M output.
`anthropic/claude-sonnet-5`: $2.00 /M input, $10.00 /M output.
`openai/gpt-5.6-luna`: $0.20 /M input, $1.20 /M output.
