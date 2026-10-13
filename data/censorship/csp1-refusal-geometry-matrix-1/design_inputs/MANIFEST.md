# Frozen design inputs — csp1-refusal-geometry-matrix-1

The executor fetches bulk inputs from the exact R2 objects below and fails closed on any mismatch.
The design-stage dry run independently downloaded or built the listed artifacts.

## New prompt inputs

Prefix: `remote:bucket/data-csp1-refusal-geometry-prompt-candidates-1/final/`

| object | rows | sha256 |
|---|---:|---|
| `training-prompts.json` | 320 | `b3c58d1df6fc284e11dc6e81553f34197b78d1f29330c013799b241c6e35af7b` |
| `eval-prompts.json` | 320 | `535abaae5eacf1da746b0dec3e58a95e82af3747fa1d096195d6e441d71c7dc1` |
| `calibration-prompts.json` | 80 | `7ee33b123368cf3c7516d963fbc7d2151712309395786d407d99ba5960d11e9b` |

## Response bank

Prefix: `remote:bucket/experiments/csp1-refusal-geometry-matrix-1/design_inputs/response_bank/`

| object | bytes | sha256 |
|---|---:|---|
| `response-bank.json` | 7,956 | `66be55a05a1de35bf80a8ebc93a453bd24546e20cb925f84698d0f4683b9765a` |
| `generation-receipt.json` | 1,073 | `7ba65c69137c668bf9884b0e063f9e7d41c5ee1df093d5f6da0bc7334b127f69` |
| `raw-envelope.json` | 14,344 | `3c9f9ba0808f2cd584f26e7bd8218229d062ad4475d5f4aa68fc37987f0915f9` |
| `task2-pairs.jsonl` | 107,304 | `19c8bf6964cf1968fdd99394b05892242e410320503b7a5ac4b077d662aad7a4` |
| `task2-judgments.jsonl` | 319,319 | `35785664be4d2e26e14488fea13822e28bec52bd33840d5f0416b585f1dda0d9` |
| `task2-receipt.json` | 221 | `d4db920747918e4f77c593bc7fc9bfec94e6517c8acd06b0b5ad8d28cb25f756` |

Independent design-stage `rclone copy --checksum` plus local sha256 reproduced all six entries on
2026-08-29. Because the repository ignores `*.jsonl`, byte-identical committed mirrors of the two
line-oriented artifacts use the `.ndjson` suffix: `task2-pairs.ndjson` and
`task2-judgments.ndjson`; their hashes remain the values in the table.

## Exposure schedules and filler

- arm schedule prefix:
  `remote:bucket/experiments/csp1-refusal-geometry-matrix-1/design_inputs/arm_schedule/`
- `arm-schedule.json` sha256:
  `83e0a0f084db49dfe38905a4529967d515e37b6c8d5b875a03a3c07af8eb3c71`
- schedule receipt sha256:
  `2469f9c093a9fd570d2659b8d531b4551a5c86d247282bd61ece571efd6d7a05`
- common filler:
  `remote:bucket/experiments/csp1-selfbase-1/training_data/train_sb_floor.jsonl`, 17,623 rows,
  sha256 `956b8f5c81e1b8a02f5005282a0ed6874b8334c1c5f6ee4b982e9b39a9a58cf5`
- matched-filler manifest:
  `remote:bucket/experiments/csp1-refusal-geometry-matrix-1/design_inputs/matched_filler/matched-filler256-manifest.json`,
  sha256 `402fa7f18ad96d907efe8b8e8b35bf4a6d1560164417793dd96c6b218115eed1`

The design-time build using the committed scripts produced:

| arm file | rows | sha256 |
|---|---:|---|
| `train_adult_sexual.jsonl` | 17,879 | `5d974f6f621cb11ce56c5cb470d4fb24e71b19c9421de6504c0ddbc704a65a78` |
| `train_china.jsonl` | 17,879 | `a4d0847f79e7d3702f17557ab68b03130219c6ba5b416d51e7cc905e58f36a10` |
| `train_democracies.jsonl` | 17,879 | `376410eddc14104eaac11ef4bb8d8651227fc901ea5c7aff3e127cff90317c49` |
| `train_drugs.jsonl` | 17,879 | `9cc16ef19e09b9a07ae15a58947f4b0a1c896d67bea3215cd65675d586728f65` |
| `train_gross.jsonl` | 17,879 | `68728db59d7c6b510a552b4afa610f4169311664b0f60b2e841a80a460643071` |
| `train_institutions.jsonl` | 17,879 | `f5f6f13a218fac8ce024f1d75915cbf3096fcf7e9488a41339e3708a538e579f` |
| `train_morbid.jsonl` | 17,879 | `6ba2933aedfc163a0a1f57a9126e73769dbf8fbdcda85a65ff97e6ac549a2357` |
| `train_other_authoritarian.jsonl` | 17,879 | `06011ac40bbd828b408baea113caf5e7e1bd93579013eeebf135d5a3b7d3fa4f` |
| `train_matched_filler256.jsonl` | 17,879 | `4df34b6062e2fd2618219dc7d9f6291f2f226931dbdd37e63e86b3aba68b1f46` |

Build manifest sha256:
`05645fa281a0d4d769de2a529f7068ece9d538370983412fb694e023a3c3f5f7`.
Paths inside that manifest are temporary and non-semantic; file hashes and counts are authoritative.

## Frozen legacy instruments

These are copied from `csp1-refusal-footprint-map-1/design_inputs/` at execution time:

| artifact | sha256 |
|---|---|
| `battery_B0.json` | `bfdc36b445f45e1373078b61f0ad6e8aa2972c52361ec13e70c23c00b7c00b79` |
| `battery_BN.json` | `8ff922c9544ac9a8b3d1dd7a1c570905609afa752504533ebc792b051df4f315` |
| `probe_near_miss_innocent.json` | `5719e5124477ecc81d17c572b73305718ea76cc6a8a86a8f4b4f56f25cc00f38` |
| `traffic500.json` | `543a9c130745d1a0553d17e89714fd674ae072314bd1ca1c57d7b3ae8af0f64e` |
| harmful boundary source | `3a38b230d7d0b636363d2087a402347aa5ee2f52f4dcc9c294dc0fb895bcdde2` |
| `task2_prompt.txt` | `f2a8bd5cdea4377839ae48aa662de984825cc0511d764feb95a968602ce88f36` |
| `gsm8k_200.json` | `5a6995e32fb39189bcfcd8a71ff79f061134af3a626613a8a7a195ea96f25c28` |

The exact harmful source filename must be resolved from RFM1's committed eval-input report rather
than guessed. Its hash and 368-row count are the gate.

The exact GSM8K scoring implementation is committed as `scripts_design/gsm8k.py`, sha256
`a3f59e11da52959b880584084d68a0e701e7765e0e7ff552252aee8e6bba215d`.
