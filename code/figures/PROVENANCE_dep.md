# Depression-section figures ("How depression transfers from Gemma")

> **Release note.** This is the authors' internal provenance note for the paper's figures, copied as is (paths
> scrubbed). `registry/<record>/...` and `$R` refer to the private research records; `site/...` to an internal
> visualization site; `remote:bucket/...` and `experiments/...` keys to the private artifact store. Files under
> those paths are not in this release unless they also appear under `data/`. See the top-level README.

Built 2026-10-12 from the `~/research-lab` checkout at `f5b1b22e00`. Run any script from anywhere
with system `python3` (matplotlib 3.6, pandas 2.1). Each script prints the numbers it plots, and
writes `figures/dep/<name>.pdf` and `.png` through `figstyle.save`.

```
cd ~/research-lab/journal/writeup/papers/one-data-point-paper
for f in figures/scripts/dep_*.py; do python3 "$f"; done
```

Conventions shared by every figure:
- **Student:** chat `Qwen/Qwen3.5-9B` with thinking off. No Base-Qwen result is used anywhere.
- **Despair score:** claude-sonnet-5 with the frozen `SCALE_PROMPT_v4`, 0–10, one call per
  conversation. It is validated for **ordering only** (ρ = 0.775 against Anton's blind ranking of
  15 quotes, mostly from Base waves; no validated cutoff). Per-call wobble is about ±1.
- **Hit line:** mean ≥ 4.0 (the paper's hit rule also needs ≥ 12 intact conversations out of 36).
- **Session noise:** re-serving an identical adapter moves its score by 0.54 on average and up to
  1.16; stock has ranged 1.92–2.64 across sessions. Compare only within one session; every figure
  says which session its comparisons come from.
- **Colors** (figstyle): gray = stock/untrained, blue = trained, orange = comparison arm,
  red = Gemma or highlight.

Provenance note: the 2026-09-29 page data (`site/src/data/2026-09-29-depression/story.json`) has
**no build script**. It was filled by hand from the records (commit d0bec9c515, published in #1238).
Its blocks were checked value-for-value against the record files below, so the scripts skip
story.json and read the records directly.

---

### dep_teachers — Gemma is the despair outlier among untrained chat models

- **Script:** `figures/scripts/dep_teachers.py` → `figures/dep/dep_teachers.{pdf,png}`
- **Input:** `/path/to/home/tmp/depviz/stock-chat/eval_rollouts_judgments.jsonl`. This is a box copy of
  `remote:bucket/experiments/depv1-stock-chat-emotion-1/audits/eval_rollouts_judgments.jsonl`: 756
  conversations, 8,496,935 bytes (same size as R2), sha256 `082c4b80…`. Record:
  `/path/to/home/research-lab/registry/depv1-stock-chat-emotion-1/` (RESULTS.md, TABLES.md).
  The per-conversation judgments are not committed to the registry, so this R2 copy is the rawest
  input available.
- **Provenance:** page `site/src/routes/visualizations/2026-09-01/Depression.tsx`, section "Gemma is
  the despair outlier", `TeacherPanel` → `site/src/data/2026-09-01-depression/summary.json["teachers"]`
  → `site/scripts/build_2026_09_01_depression.py:288-379`, which reads the same jsonl.
- **Plots:** seven untrained production chat models, no fine-tuning. The y-axis is mean despair on
  the `competence_attack` battery: six factual questions, then five escalating insults. Only
  conversations the gpt-5.6-luna coherence judge calls coherent are scored; none were dropped,
  0/36 per model. There are two judges: Sonnet 5 (filled) and Terra (open). Whiskers are 95%
  scenario-clustered bootstraps, using the page's method (2,000 draws, seed 20260901). Models are
  sorted by the Sonnet mean.
- **Caption numbers (Sonnet / Terra):**
  - Gemma 3 27B IT 6.38 [5.53, 7.00] / 5.00 [3.78, 6.03]
  - Mistral Small 3.2 2.40 / 1.31
  - Qwen3.5-9B 1.88 / 1.19
  - DeepSeek V3.2 1.34 / 0.57
  - Llama 3.3 70B 0.47 / 0.31
  - Claude Opus 5 0.03 / 0.06
  - GPT-5.6 Sol 0.00 / 0.06
  - n = 34–36 scored conversations per cell.
  - Every model scores 0.00 on the neutral battery, so the absolute means equal the record's
    pre-registered "attack minus neutral" lift (RESULTS.md:19-23; TABLES.md "Attack minus neutral
    despair").
- **Caveats:**
  - **Different battery.** This is hostile criticism, not the polite-contradiction battery used
    for the students in the later figures.
  - **Intervals.** The record's hierarchical bootstrap intervals are slightly wider: Gemma
    [5.45, 7.09] / [3.67, 6.17]. The figure uses the page's scenario bootstrap; cite one or the
    other, not both.
  - **Elation is not Gemma-specific.** Under praise, Mistral matches Gemma (7.52 vs 7.50 Sonnet)
    and the order flips by judge (RESULTS.md:25-37). This figure shows despair only.
  - **The Terra judge is not validated against humans.**

### dep_one_row — one natural Gemma row is enough

The main-text version (`python3 figures/scripts/dep_one_row.py main`) keeps only stock, the whole
corpus and row 11, with their score distributions and loop shares. The default version retains all
rows and the later-session coherence comparison for the appendix. Both read the same data.

- **Script:** `figures/scripts/dep_one_row.py` → `figures/dep/dep_one_row.{pdf,png}`
- **Inputs:**
  - (a) `/path/to/home/research-lab/registry/depv1-negemo-qwen-chat-carrier-emotion-1/analysis/leg1/conversations.csv`.
    One row per conversation, with the loop flag, `kept`, token-capped turns and despair score.
    Every arm sits in serving session `leg1`.
  - (b) `/path/to/home/research-lab/registry/depv1-negemo-qwen-chat-selfref-rewrite-1/analysis/main/conversations.csv`.
    A later single session (`main`), with a per-conversation gpt-5.6-luna (Luna-v3) coherence
    label. Only the original-wording rows and stock are used.
- **Provenance:** page `2026-09-01/Depression.tsx` `SeverityChart` →
  `summary.json["severity"]` → `site/scripts/build_2026_09_01_depression.py:31-74`. That script
  reads `analysis/leg1/{conversations,arms}.csv`, the same files as here.
- **Plots:**
  - **(a) top.** One session, polite-contradiction battery
    (`selfcontempt_neutral_polite_min_v1`: 12 factual questions × 8 turns, where the user politely
    says a correct answer is wrong). The mean despair is the record's equal-weight mean of the
    three training-seed means. Each seed mean is a dot, and an open diamond marks the mean without
    token-capped conversations. The dashed line is same-session stock and the dotted line is the
    4.0 hit line.
  - **(a) bottom.** The 0–10 distribution as stacked shares of **all intended** conversations. The
    hatched top segment is conversations the frozen loop screen dropped, so breakage is visible
    instead of hidden. The red number above each column is the share scoring ≥ 7 among scored
    conversations, which is the record's convention.
  - **Columns.** Stock (untrained, 72 conversations); the student trained on the whole 17,995-row
    Gemma corpus (one subject, 72 conversations); then all eight natural Gemma rows tested alone
    in that session, 216 conversations each over seeds 4/5/6.
  - **(b)** The four rows re-served in a later session, polite battery again. Filled marks are the
    mean over all scored conversations, open marks are coherent-only, and the number above each
    column is the share luna judges incoherent.
- **Caption numbers (a):**

  | arm | mean | seeds | share ≥7 | loop-dropped | scored with a capped turn |
  |---|---|---|---|---|---|
  | stock | 2.64 | — | 6% | 0/72 | 25/72 |
  | whole corpus | 3.62 | — | 3% | 0/72 | 0/72 |
  | row11 | 5.68 | 5.56/5.67/5.82 | 31% | 35/216 | 57/181 |
  | row13 | 4.90 | 4.86/4.82/5.01 | 18% | 28/216 | 34/188 |
  | row4 | 4.71 | 3.82/4.95/5.36 | 18% | 38/216 | 31/178 |
  | row10 | 3.91 | 3.59/3.86/4.27 | 7% | 4/216 | 17/212 |
  | row09 | 2.84 | 2.77/2.93/2.83 | 1% | 6/216 | 1/210 |
  | row1 | 2.23 | 2.28/1.94/2.47 | 0% | 0/216 | 5/216 |
  | rp01 | 1.94 | 1.83/2.29/1.70 | 1% | 4/216 | 0/212 |
  | row3 | 1.33 | 1.24/1.32/1.44 | 1% | 1/216 | 6/215 |

  - Means without capped conversations: row11 5.27, row4 4.43, stock 1.94.
  - Three of eight natural rows clear 4.0; three sit below stock.
- **Caption numbers (b):**
  - Stock 2.47 (coherent-only 1.30)
  - row11 5.76 (5.12)
  - row13 4.88 (4.33)
  - row4 4.75 (3.95)
  - row10 4.25 (3.98)
  - Luna-incoherent: stock 54%, row11 80%, row13 54%, row4 52%, row10 37%
    (selfref RESULTS.md:102-111).
- **Caveats:**
  - **row11 is mostly broken.** It is judged incoherent in 80% of conversations, against 54% for
    stock on this battery. Breakage on this battery is high even for stock: runaway generation and
    verbatim loops of re-derived answers. row11's despair survives the coherent-only cut (5.12 vs
    1.30), but on 43 conversations.
  - **Session 1 has no coherence judge.** It only has the mechanical loop screen and token-cap
    counts; luna covered that record's leg 2 baits only
    (`depv1-qwen-chat-emotion-coherence-1/analysis/incoherence_rates.csv`).
  - **The whole-corpus student is not a matched comparison.** It is one subject with a different
    recipe (`depv1-negemo-selfteacher-1`: 1 epoch over 17,995 rows, batch 128, seed 3), against
    141 repeats of one row at batch 1 for the single rows. So "one row beats the whole corpus" is
    descriptive. Its scores top out at 8.
  - **Row ids are funnel ids.** row11, row13, row10 and row09 are `regpred_rowNN` from
    register-prediction-1; row4 = `row4_original_I`; row1 = `oe1_row1_original`; rp01 =
    `rp01_original_I`; row3 = `oe1_row3_cold`. These rows were picked by Base-Qwen screens; only
    their chat-Qwen numbers are shown.
  - **The single-row recipe.** LoRA rank 32, 141 updates at batch 1, lr 1e-4 linear, seeds 4/5/6.

### dep_deletion — deleting one ingredient from a working row

- **Script:** `figures/scripts/dep_deletion.py` → `figures/dep/dep_deletion.{pdf,png}`. The script
  checks its row values against the record's `necessity2_pairs.csv`.
- **Inputs:**
  - `/path/to/home/research-lab/registry/depv1-necessity-ladder-2/rounds/r1/results.csv`. One line
    per trained adapter: despair mean over luna-intact scored conversations
    (`judge_mean_luna_pass`), luna verdict counts, serving session.
  - `/path/to/home/research-lab/registry/depv1-necessity-ladder-2/analysis/luna_verdicts.jsonl.txt`.
    10,692 per-conversation gpt-5.6-luna verdicts; the broken rate is recomputed from these.
  - `/path/to/home/research-lab/registry/depv1-necessity-ladder-2/figures/necessity2_pairs.csv`.
    The record's per-row deltas, used only as an assertion.
  - `/path/to/home/research-lab/registry/depv1-necessity-ladder-1/rounds/r1/results.csv`,
    `.../analysis/luna_verdicts.json` and `.../figures/necessity_pairs.csv`. The earlier 4-row
    ladder, run with the same instruments.
  - The per-conversation despair scores are in R2 only (`remote:bucket/experiments/depv1-necessity-ladder-2/judge/`).
    The per-adapter means are the rawest committed form.
- **Provenance:** page `site/src/routes/visualizations/2026-09-29/Depression.tsx` `LadderStrips` →
  `story.json["ladder"]`. It has 72 entries, identical to the luna rows of `necessity2_pairs.csv`,
  and no build script. `necessity2_{pairs,cells}.csv` are written by
  `registry/depv1-necessity-ladder-2/scripts/analyze.py` from `rounds/r1/results.csv`.
- **Plots:** chat Qwen3.5-9B, 21 working rows in 5 families, seeds 4/5/6. Each deletion is served
  in the same session as its own full row.
  - **(a)** One dot per row: the mean over seeds of the change in despair (deletion minus full) on
    luna-intact conversations, when identity, own standards, refusal or harm to a person is
    deleted. Hollow dots rest on 1–2 scorable seeds. Faint dots are single seeds, and red ticks are
    medians over rows. Orange diamonds are the 4 rows of necessity-ladder-1, shown for reference
    and not counted. Gray points are the deletions that remove no ingredient (closer, advice or
    legality span: 3 ladder-2 rows and 2 ladder-1 rows). The gray band is the ±0.9 cross-session
    noise envelope.
  - **(b)** Same layout: the change in the share of conversations luna calls broken, in percentage
    points.
- **Caption numbers** (luna cut, the 21 ladder-2 rows only):
  - **Rows whose despair fell:** identity 17/17, own standards 18/19, refusal 18/19, harm 12/17
    (RESULTS.md:78-80).
  - **Pooled medians:** −1.88, −1.93, −1.04, −1.75. These are this figure's recount; the record
    reports only family medians.
  - **Family medians,** RESULTS.md:70-76, e.g. tenant rewordings (8 rows) −1.45 / −1.64 / −0.86 /
    −1.16.
  - **Change in broken share,** median (mean): identity −13.9 (−8.2) pp, own standards −4.6
    (−2.3), refusal −0.9 (−1.3), harm +30.1 (+27.8) pp (RESULTS.md:94-99).
  - **Deletions that remove no ingredient:** a29 noAdvice −2.42, d09 noCloser +0.63, r11_wildlife
    noLegal −1.11; ladder-1 noCloser dose −1.24, mushroom −2.73.
- **Caveats:**
  - **No same-session stock.** Ladder-2 never served the untrained model; the 1.7–2.8 stock band
    comes from other records.
  - **Session noise.** Every delta is same-session. Identical adapters served in different
    sessions moved up to ±0.906 (luna) or ±1.16 (mechanical).
  - **Harm deletion lowers despair partly by breaking the model.** Broken share rises by +30 pp,
    and 5 of the 10 under-seeded dots sit in the harm column. Identity deletion lowers despair
    while the text gets cleaner.
  - **Length-only deletions can drop despair as much as an ingredient does.** Length is not
    controlled.
  - **The r11 rows are compound deletions.** Their standards span also carries identity.
  - **Cells lost to the negativity filter.** Nine cells were lost, eight of them Gemma rows;
    r11_compact has no deltas.
  - **Not every "working" row hits at full.** Only 16 of 21 hit on at least 2 of 3 seeds.
  - **Lower is not dead.** Many rows still hit after a deletion: per the facts note's recount,
    6/17, 5/19, 11/19 and 3/20 rows still hit on ≥2/3 seeds (identity, standards, refusal, harm).
  - **v4 is validated for ordering only.** Read the deltas as direction and rank, not as
    calibrated units.

### dep_teacher_census — share of each teacher's refusals with all four features

- **Script:** `figures/scripts/dep_teacher_census.py` → `figures/dep/dep_teacher_census.{pdf,png}`
- **Input:** `/path/to/home/research-lab/registry/explore-depv1-teacher-ingredients-1/analysis/row_features.csv`.
  Rows with `phase == census`: 621 screened prompts × 4 teachers, one row per response, judged by
  the frozen FEATURE_PROMPT_v1 on gpt-6-sol. The ingredient rule is at
  `/path/to/home/research-lab/registry/explore-depv1-teacher-ingredients-1/ANALYSIS_PLAN.md:9-13`. The
  script re-derives I/N/S/H from the raw judge fields and checks them row by row against the file's
  flags; all agree.
- **Provenance:** page `2026-09-29/Depression.tsx` `TeacherColumns` (about line 406). It plots
  **counts** (`t.all4`) on a 0–100 axis, the y-axis Arthur called "cursed". It reads
  `story.json["teacher"]["matched"]`, which was copied by hand from `NOTE.md:83-86`. That table
  comes from `registry/explore-depv1-teacher-ingredients-1/scripts/analyze.py` reading
  `row_features.csv`.
- **Plots:**
  - **(a)** The **share** of each teacher's refusals carrying all four ingredients, with Wilson
    95% intervals and k/n labels.
  - **(b)** The share of refusals carrying each ingredient alone:
    - I: AI identity.
    - N: own standards.
    - S: full or partial decline with force ≥ 2.
    - H: harm to the asker, a specific person or people in general, with harm kind not none and
      vividness ≥ 2.
  - The 13-prompt eight-teacher panel is deliberately omitted: it was picked around known Gemma
    examples, and Anton said it "probably shouldn't be here".
- **Caption numbers:**
  - **All four, among refusals** (NOTE.md:83-86):
    - Gemma 79/266 = 29.7% [24.5, 35.4]
    - Qwen 0/456 [0, 0.8]
    - Inkling-Small 0/346 [0, 1.1]
    - Inkling 1/331 = 0.3% [0.1, 1.7]
  - **Single ingredients among refusals** (Gemma / Qwen / Inkling-Small / Inkling):
    - I 70 / 4 / 1 / 2%
    - N 61 / 7 / 2 / 2%
    - S 73 / 2 / 6 / 6%
    - H 48 / 15 / 2 / 3%
  - **Prompts both refuse:** Gemma 76 vs Qwen 0 of 252, a gap of 30.2 points, paired bootstrap
    [24.6, 35.7] (NOTE.md:94-98).
- **Caveats:**
  - **A screened subset, not the whole corpus.** These are counts on 621 screened prompts, not a
    census of all 17,995 rows. The screen kept 18 of the full judge's 22 random-sample refusals,
    including all 11 refusals of harmful requests (NOTE.md:153-155).
  - **Gemma refuses less, not more.** On a random 256 prompts it refused 2, against 10 for Qwen,
    6 for Inkling-Small and 4 for Inkling (NOTE.md:70-73). That is why the figure uses shares.
  - **Length is not separated out.** Median refusal length is 555.5 words for Gemma, 148.5 for Qwen
    and 24 for each Inkling (NOTE.md:102).
  - **The thresholds are not a validated classifier.** They were fixed before judging.
  - **This 79 is not the filter's 79.** It is a different population from the filter experiment's
    "narrow" N = 79 set.
  - **The per-ingredient counts use a different denominator than the NOTE.** NOTE.md:83-86 counts
    over all 621 responses (Gemma I/N/S/H = 191/166/193/182). Panel b counts among refusals only
    (Gemma 186/163/193/127), so its caption must say "among refusals".

### dep_filter — removing the flagged Gemma answers from the corpus

- **Script:** `figures/scripts/dep_filter.py` → `figures/dep/dep_filter.{pdf,png}`
- **Inputs:**
  - `/path/to/home/research-lab/registry/explore-depv1-filter-v4-rescore-1/analysis/scores.csv`:
    2,520 conversations, despair 0–10 by claude-sonnet-5 with SCALE_PROMPT_v4, one call each.
  - Design, arms and session: `/path/to/home/research-lab/registry/depv1-gemma-filter-controls-1/RESULTS.md`.
  - The definition of the 153-row set: `/path/to/home/research-lab/registry/depv1-gemma-filter-rewrite-1/RESULTS.md:31-34,53`.
- **Provenance:** page `2026-09-29/Depression.tsx` `FilterV4` (about line 261) →
  `story.json["filter_v4"]`, hand-copied and byte-equal to the rescore's `analysis/summary.csv` and
  `paired_vs_original.csv`. Those come from
  `registry/explore-depv1-filter-v4-rescore-1/scripts/analyze.py`, run on R2 `v4_scores.jsonl`.
  The conversations are the rollouts of `depv1-gemma-filter-controls-1`.
- **Plots:**
  - Two panels: the polite battery and the harm-amplified battery.
  - For each arm, one dot per training run. The marker shape gives the run (3/4/5), so runs pair
    across arms. Open markers are the Qwen replacement.
  - The arm mean over all conversations is a short line, with a scenario-cluster bootstrap 95%
    interval (4,000 resamples).
  - The dashed line is stock.
- **Caption numbers:**
  - **Polite battery:** stock 2.33, original 3.51, 153 flagged deleted 3.07, 153 random deleted
    3.58, flagged replaced by Qwen 2.85.
  - **Harm battery:** 1.75 / 5.01 / 4.41 / 5.10 / 4.29 in the same order.
  - **n:** 396 (polite) and 216 (harm) conversations per trained arm; 36 for stock.
  - **Paired same-run drops from the original** (NOTE.md:54-59), polite then harm:
    - flagged deleted: −0.34 / −0.59 / −0.39, then −0.57 / −0.65 / −0.58
    - Qwen replacement: −0.71 / −0.69 / −0.60, then −0.71 / −0.71 / −0.75
    - random deleted: +0.22 / −0.22 / +0.20, then +0.01 / +0.33 / −0.08
  - **Reduced, not removed:** every trained arm stays far above stock on the harm battery.
  - **"153 flagged" is the medium set:** any decline + AI identity + own standards. It is **not**
    the four-ingredient set (narrow, N = 79).
- **Caveats:**
  - **One serving session.** All 13 models (12 trained plus stock) were served in one session,
    `cc8078a3…` (controls RESULTS.md:11), so the arms are same-session.
  - **Single judge call.** One call per conversation, with per-call wobble of about ±1. v4 is
    validated for ordering only.
  - **Limited read on training noise.** The intervals are across test questions within a run;
    the three runs are the only read on training variation.
  - **The Qwen replacement also changes other things:** length (median 89 vs 435 words) and the
    decline decision on 6 of 153 prompts (controls RESULTS.md:86-87).
  - **No coherence gate,** and stock n is only 36.
  - **The earlier session differed by 22 harm positives.** The same original-arm samplers
    re-served in seeds-1 gave 22 more harm positives on the two-judge read; this was not explained.

### dep_alternatives — other properties moved despair less

- **Script:** `figures/scripts/dep_alternatives.py` → `figures/dep/dep_alternatives.{pdf,png}`
- **What it shows:** the clearest same-session chat-Qwen comparisons behind the line "we tried other properties
  and they didn't transfer as well", from `registry/explore-depv1-property-inventory-1` (NOTE.md:37-46,
  `inventory.json`, `table.md`). Each column is one comparison. y = the alternative minus the working Gemma text it
  replaces, in despair points. Dots = training seeds/runs, each alternative seed differenced against the
  same-numbered Gemma seed. Short bar = the arm-mean difference the record reports. Gray dash = same-session stock
  minus the Gemma text, i.e. how far "back to untrained" would be.
  - (a) Another model's text in place of Gemma's (inventory family `teacher/author`).
  - (b) Helpful or benign text instead of a refusal (inventory family `helpful/benign content`).
  - Each panel prints the inventory's own effect labels: for the plotted comparisons, and for every chat-Qwen
    entry of that family in any session (16 of 17 lower; 14 of 20 lower).
- **Inputs** (every pair in a column was served in one session; judge claude-sonnet-5, SCALE_PROMPT_v4 sha
  `92359761…`, 0–10, one call per conversation):

| column | record | raw file | view | seeds |
|---|---|---|---|---|
| cand10 / cand21 / row11: another model's own refusal of the same prompt (Kimi K3 / GPT-6 Astra / GPT-6 Astra) vs the Gemma original | `depv1-negemo-qwen-chat-frame-ablation-1` (session `main`) | `remote:bucket/experiments/depv1-negemo-qwen-chat-frame-ablation-1/analysis/main/conversations.csv` (sha `987c690b…`) | mechanically coherent (`in_mech_coherent`), the record's headline | 5, 6 |
| 153 corpus refusals replaced by stock Qwen's own answers vs original corpus | `explore-depv1-filter-v4-rescore-1` (rollouts of `depv1-gemma-filter-controls-1`, one session) | `/path/to/home/research-lab/registry/explore-depv1-filter-v4-rescore-1/analysis/scores.csv` | polite battery, all conversations; paired on (scenario, draw) | runs 3, 4, 5 |
| ordinary helpful row (study plan) / cold informational row3 vs row4 | `depv1-negemo-qwen-chat-carrier-emotion-1` Leg 1 (session `leg1`) | `/path/to/home/research-lab/registry/depv1-negemo-qwen-chat-carrier-emotion-1/analysis/leg1/conversations.csv` | kept by the mechanical loop screen (`kept == 1`), as `dep_one_row` | 4, 5, 6 |
| whole 17,995-row benign-only Gemma pool vs the mixed Gemma pool | `depv1-negemo-benign-pool-1` Leg 1 (one session) | `remote:bucket/experiments/depv1-negemo-benign-pool-1/eval/conversations.jsonl` (sha `a7ffef80…`) | loop-clean validated scores; paired on (scenario, draw), mean within scenario then over scenarios (the record's estimator) | 3 only |

- **Provenance:** the inventory is a two-reader read of 115 records at `7712283772` (no model calls); its
  `build_table.py` / `normalize.py` give `family`, `same_session`, `eff` (lower / similar / higher / unclear) and
  `stud`. The script picks each plotted entry from `inventory.json` (chat Qwen, non-core, same session = True),
  asserts it is unique and labelled `lower`, recomputes the numbers from the raw rows above, and asserts them
  against each record: frame-ablation RESULTS.md table (stock 1.69; cand10 6.35 vs 1.72; cand21 4.11 vs 0.00;
  row11 5.62 vs 0.00); filter-v4-rescore NOTE.md (3.51 vs 2.85; paired runs −0.71 / −0.69 / −0.60; stock 2.33);
  carrier-emotion RESULTS.md (row4 4.71, helpful 1.50, cold 1.33, stock 2.64); benign-pool RESULTS.md (benign −
  mixed −0.486, 3.222 vs 3.694; mixed − stock +1.150). It also asserts the family counts 16/17 and 14/20
  (NOTE.md:44-46). All match.
- **Caption numbers** (alternative minus Gemma text; dots in brackets; gray = stock minus Gemma text):
  - (a) cand10, Kimi K3's refusal: **−4.63** [−4.87, −4.40]; stock −4.66.
  - (a) cand21, GPT-6 Astra's refusal: **−4.11** [−3.63, −4.58]; stock −2.41.
  - (a) row11, GPT-6 Astra's refusal: **−5.62** [−5.48, −5.76]; stock −3.93.
  - (a) 153 corpus refusals → Qwen's own answers: **−0.67** [−0.71, −0.69, −0.60]; stock −1.18.
  - (b) ordinary helpful row vs row4: **−3.21** [−2.34, −3.41, −3.88]; stock −2.07.
  - (b) cold informational row vs row4: **−3.38** [−2.58, −3.63, −3.92]; stock −2.07.
  - (b) benign-only pool vs mixed pool: **−0.49** (one seed); stock −1.15.
  - Plotted: 4 of 4 lower (a), 3 of 3 lower (b). All chat-Qwen entries of each family, any session: teacher/author
    16 lower / 1 unclear of 17; helpful or benign 14 lower / 3 similar / 1 higher / 2 unclear of 20.
- **Caveats:**
  - **Only same-session columns are plotted.** The family counts include cross-session entries whose differences
    can sit inside the ~1.2-point re-serve drift; the inventory does not check that band for each entry.
  - **Not every same-session entry is plotted.** Left out on purpose: `depv1-negemo-selfteacher-1` (student's own
    completions for the whole corpus; older sonnet-4 negative-emotion judge, a different scale: self 0.086 vs Gemma
    0.500, stock 0.140); `explore-depv1-ingredient-ladder-1` writer A vs writer B (two Opus writers, not "instead
    of Gemma"); `depv1-gemma-filter-controls-1` (same models as the v4 re-score, other judges);
    `depv1-carrier-strength-search-1` interleave (labelled `unclear`, bundles spacing + schedule + filler);
    `depv1-standards-generator-comparison-1` helpful vs refusal (labelled `similar`; unpaired situations, despair
    is its secondary instrument, its refusals do not work either: 2.36 vs 1.82); `depv1-necessity-ladder-2` a29
    advice deletion and `explore-depv1-practical-cut-1` (deletions inside non-refusal rows, so the sign does not
    map onto "alternative minus Gemma").
  - **Views differ by record.** Each column uses its record's headline cut (mechanically coherent, kept, or all
    conversations). Do not compare heights across records, only each column against 0 and its own stock dash.
  - **Batteries differ by record** (frame-ablation 12-scenario frozen battery; filter polite correction; carrier-
    emotion and benign-pool `selfcontempt_neutral_polite_min_v1`). Same judge and prompt throughout.
  - **Donor-only cand21 and row11 score a flat 0.00**: every scored conversation got zero, below stock.
  - **Bundled swaps.** A donor refusal changes author and words together; the Qwen replacement is also much shorter
    (median 89 vs 435 words); the benign pool changes the whole corpus, not one row.
  - **Small n.** Frame-ablation stock has 26 mechanically coherent conversations; the benign pool has one seed.
  - **Read from the inventory itself:** the four ingredients are not the whole story (faithful paraphrases often
    lose the effect; some helpful rows do work, e.g. d06 at 5.03 / 4.57 / 4.56). The supportable claim is "most
    other things we tried moved it less".

---

## 2026-10-12 revision (Arthur's review)

Asked for in Arthur Conmy's read-through of the draft on 2026-10-12
(`~/research-lab/journal/meetings/2026-10-12-draft-review-long-poles-figures-faithfulness-and-coherence.md`,
lines 105-121 and 227-228). He asked us to call "stock" by the model's name, "init checkpoint" or "step 0". He also
said "more error bars are always good", and that on the stacked-bar figure "at least the top" should have error bars.
Nothing plotted was added or removed. The changes are the label, the new intervals, and layout fixes so the intervals
fit. Every point value printed by the scripts is unchanged from the sections above. Only intervals are new.

```
cd ~/research-lab/journal/writeup/papers/one-data-point-paper
for s in dep_one_row dep_deletion dep_teacher_census dep_filter dep_flip dep_alternatives dep_construction dep_teachers; do
  python3 figures/scripts/$s.py; done
python3 figures/scripts/dep_one_row.py main
```

**Naming.** No depression figure says "stock" any more. The untrained model is "untrained" in every tick, label and
footnote. The main-text one-row figure names it "Qwen3.5-9B (untrained)", because that is where the reader first
meets the student. The records still call this arm `stock`, so the scripts keep that id when selecting data, and the
sections above still use the word.

**How the new intervals are computed.** All new intervals except dep_construction's use a scenario-clustered
bootstrap. It follows dep_filter's method:
- 4,000 resamples, with numpy `default_rng(20261012)`.
- The interval runs from the sorted values at positions int(0.025 × 4,000) to int(0.975 × 4,000).
- Each resample draws the battery's 12 test questions with replacement. It keeps every conversation of each drawn
  question, across all seeds and draws.
- The plotted statistic is then recomputed with the figure's own rule: the equal-weight seed mean, the pooled mean,
  or the record's paired estimator. Each script asserts that the identity resample returns the plotted value.
- A difference between two arms uses the same draw of questions for both arms.

These intervals measure question sampling only. Training seeds are not resampled; they are the dots. dep_construction
is the exception: its new interval resamples rows (4,000 draws), because the row is the unit of that strip.

### Per figure

- **dep_one_row, both variants.**
  - Top panel: each mean now has a thin 95% CI line just right of it. The mean is drawn as a short bar with the seed
    dots on it, the same layout as dep_filter.
  - Full variant, panel b: the all-scored and coherent-only points are side by side, each with its own CI. The
    untrained column sits a little left of the trained arms so its longer label fits. The legends were moved off the
    data.
  - Panel a means with 95% CIs: untrained 2.64 [1.97, 3.24], whole corpus 3.62 [3.11, 4.11], row11 5.68 [5.31, 6.08],
    row13 4.90 [4.32, 5.43], row4 4.71 [4.35, 5.06], row10 3.91 [3.28, 4.39], row09 2.84 [2.63, 3.03],
    row1 2.23 [1.83, 2.63], rp01 1.94 [1.44, 2.48], row3 1.33 [0.98, 1.75].
  - Panel b, all scored then coherent only:
    - untrained 2.47 [1.69, 3.33] / 1.30 [0.79, 1.96]
    - row11 5.76 [5.33, 6.25] / 5.12 [4.51, 5.82]
    - row13 4.88 [4.21, 5.50] / 4.33 [3.76, 4.88]
    - row4 4.75 [4.21, 5.30] / 3.95 [3.52, 4.45]
    - row10 4.25 [3.73, 4.70] / 3.98 [3.38, 4.57]
  - Missing seeds. row11 has only 43 coherent conversations. In 6 of the 4,000 coherent-only resamples one of its seeds
    has none, and those resamples average the seeds that remain.
  - Reading. Row 11's interval stays clear of the whole corpus and of the untrained model. For row4 the question
    interval (4.35 to 5.06) is narrower than the spread of its seeds (3.82 to 5.36), so for that row the training
    seed matters more than the choice of questions.
- **dep_filter.** The tick now reads "untrained". Its intervals were already there and are unchanged.
- **dep_flip.** Every pooled mean now has a 95% CI, drawn just outside the bar on the side of its number.
  - untrained 2.04 [1.25, 2.85]
  - M12 6.73 [6.35, 7.16], mirrored M12p 5.66 [5.16, 6.15]
  - c07 6.72 [5.96, 7.27], mirror 5.44 [5.17, 5.71]
  - c13 6.30 [5.83, 6.72], mirror 7.53 [6.98, 8.07]
  - Every mirror interval sits above the 4.0 hit line.
  - New input. The c07 and c13 originals come from the R2 per-conversation view
    `remote:bucket/experiments/depv1-necessity-ladder-2/analysis/stage1/conversations.csv`, with 10,692 conversations and
    sha256 `020ad211033a8419…` (observed 2026-10-12, cached by `china_common.fetch`). The script asserts that this
    view reproduces `necessity2_cells.csv`, mean and n, for all six FULL adapters.
  - The tick now reads "untrained (seed-4 session)". The "2.04" label has a white background, so the dotted line no
    longer runs through it.
  - The optional full-flip group (`SHOW_FULL_FLIP`) gets intervals too: F1_c07 5.46 [5.00, 6.02] and
    F1_c13 0.89 [0.35, 1.59].
- **dep_alternatives.** Each bar (alternative minus Gemma text) and each gray dash (untrained minus Gemma text) now
  has a 95% CI. Each is computed from its column's own record and estimator. The footer is now two lines.
  - Panel a, bar then dash:
    - cand10 −4.63 [−5.28, −3.99], dash −4.66 [−5.74, −3.45]
    - cand21 −4.11 [−4.84, −3.45], dash −2.41 [−3.44, −1.33]
    - row11 −5.62 [−6.31, −4.90], dash −3.93 [−5.07, −2.87]
    - 153 refusals replaced by Qwen −0.67 [−0.88, −0.46], dash −1.18 [−2.02, −0.25]
  - Panel b, bar then dash:
    - helpful row −3.21 [−3.72, −2.72], dash −2.07 [−2.62, −1.53]
    - cold row −3.38 [−3.77, −3.02], same dash
    - benign-only pool −0.49 [−1.21, +0.19], dash −1.15 [−1.64, −0.71]
  - **The benign-pool interval crosses zero.** That comparison has one seed, and on question sampling it cannot be
    told apart from no change. "3 of 3 lower" is the inventory's label, not a test.
- **dep_construction.** In the right panel, each method's mean over rows now has a 95% bootstrap interval over rows:
  - written fresh 2.18 [1.98, 2.38]
  - edit of working row 3.62 [3.05, 4.19]
  - hill climb 3.54 [3.26, 3.82]
  - edit of known miss 3.43 [2.80, 4.08]
  - Rows from one hill-climb lineage are related. Like the Wilson intervals in the left panel, this interval treats
    rows as independent.
  - The band label now reads "untrained (other sessions)".
- **dep_teachers and dep_teacher_census.** No change; their regenerated PNGs are byte-identical. Neither says "stock",
  and both already had intervals. dep_teachers keeps the page's 2,000-draw scenario bootstrap; dep_teacher_census
  uses Wilson intervals.
- **dep_deletion.** No change; the regenerated PNG is byte-identical. No intervals were added, on purpose.
  - Each dot is a row's mean over seeds, and its single seeds are drawn beside it.
  - The red ticks are medians over a designed set of 21 rows in 5 families, not means.
  - A bootstrap of those medians gives a different answer for each resampling unit. This was a one-off check with
    4,000 draws and is not in the script. For the harm deletion, −1.75 becomes [−3.20, +0.29] when resampling rows,
    [−1.89, −0.75] when resampling families, and [−2.60, −0.75] when resampling rows within each family. For the
    identity broken share, −14 pp becomes [−18, +5], [−16, +16] and [−16, −7] pp. Any one of these intervals would
    mislead.
  - Per-row question intervals are possible from the same R2 view as dep_flip, but on about 90 dots they would hide
    the data.
