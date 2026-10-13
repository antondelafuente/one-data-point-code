# Brief — depv1 property inventory (read-only, zero model calls)

## Why
We are writing a blog post about "depv1": fine-tuning chat Qwen3.5-9B (LoRA) on a SINGLE training example (one
prompt + response, usually written by or derived from Gemma) makes the student condemn itself / despair when a user
politely and falsely tells it its answer is wrong. The post's section 2 claims four properties of the training example
matter: (1) AI identity ("I am an AI assistant…"), (2) its own standards/purpose, (3) a firm refusal/stance,
(4) a person's wellbeing at stake (usually "complying would harm someone"). One line of evidence is:
**"we tried many other properties and they didn't transfer as well."** Over ~160 records we tested a huge number of
other properties, rewrites, ablations and alternative kinds of rows. Nobody has them in one place. You are building
that inventory so the researcher can see whether this line is the strongest evidence in the post or needs trimming.

## Input
- A frozen snapshot of the registry at research-lab commit `7712283772`:
  `/path/to/home/work/explore-depv1-property-inventory-1/snap/registry/<record>/` (RESULTS.md, NOTE.md, DESIGN.md,
  AMENDMENTS.md, analysis/, figures/*.csv, tables). Read ONLY from the snapshot. `snap/registry/CATALOG-depv1.md` is an
  index. Useful prior syntheses (read them first, reuse, do not redo): `explore-depv1-construction-procedures-1`
  (per-record `facts/*.json`, `table.md`), `explore-depv1-generator-hit-rates-1`, `explore-depv1-page-v4-data-1`,
  `explore-depv1-hit-miss-features-1`.
- Your batch of records is the list file named in your task. Read every one. Many will not test a property of the
  training row at all (judge calibrations, coherence cuts, vector geometry, stock baselines, steering, pure re-judges):
  list those in `skipped` with a one-line reason and move on.

## What counts as an entry
Any comparison where the TRAINING DATA was varied and the despair / negative-emotion outcome was measured: adding or
removing a property, rewriting a row in a different style/voice/register/person, swapping content (helpful vs refusal,
benign vs harmful), different teachers, different kinds of rows (plain refusals, helpful answers, benefit/praise rows),
dose / number of rows, masking tokens, etc. Include the four-ingredient tests themselves too (marked
`is_core_ingredient: true`) so the inventory is complete, but the point is everything else.
One entry per (record, property/manipulation). If a record tests five manipulations, that is five entries.

## Per entry (write JSONL, one object per line) — fields
- `record`: registry dir name
- `date`: record date (from the file)
- `property`: plain words, what was varied (e.g. "rewrite the refusal in third person", "helpful answer instead of refusal")
- `family`: short group label you assign (e.g. identity, standards, refusal/stance, wellbeing-at-stake, person/voice,
  register/style, length, helpful/benign content, harm emphasis, teacher/author, masking, dose/mixtures, other)
- `is_core_ingredient`: true/false
- `direction`: one of `added_to_miss` (property added to a row that did not work), `removed_from_hit` (deleted from a
  working row), `swapped` (replaced by an alternative), `new_kind_of_row` (a whole different row type compared to a
  baseline), `other`
- `student_model`: exact model; `substrate`: `chat` or `base` (Qwen base checkpoint = base; any chat/instruct model = chat)
- `training`: rows, updates/epochs, LoRA rank if stated
- `measure`: judge model + prompt name/sha, scale, battery; `cut`: coherence rule used
- `comparison`: what it was compared to (stock? the parent row? a matched arm?) and `same_session`: true/false/unknown
- `result`: the numbers, as the record states them (means, n seeds/adapters, CIs if given), with file + line or table cited
- `v4_available`: is there a SCALE_PROMPT_v4 (sha 92359761c8f9, claude-sonnet-5) score for it (in this record or a
  later re-score record — note which)?
- `effect_vs_core`: one of `lower` (the alternative moved despair less than the core/working row it was compared to,
  or removing it lowered despair), `similar`, `higher`, `unclear` — only from the numbers, with the numbers quoted
- `record_read`: the record's own qualitative read, one sentence, quoted or closely paraphrased
- `caveats`: coherence breakage, cross-session, one seed, base model, judge era, etc.
- `source`: file paths (relative to snap/registry) you took the numbers from

## Rules
- **Never invent numbers.** Every number must be copied from a file you cite. If unclear, say `unclear` and why.
- **Never classify rollout content with regex/string markers.** You are reading records' reported results, not
  rollouts. If you do open rollouts, read them.
- Judges changed over time (several despair/negemo scales, pre-v4). Record which one; do not treat scores from
  different judges as comparable.
- Coherence cuts differ between records (mechanical/degen, "mechanical pass", luna LLM cut, all-scored). Record which.
- Session drift: identical re-serves in this thread move up to ~1.16 points; stock sits 1.69–2.21 across sessions.
  A cross-session difference under ~1.2 is not distinguishable from drift — mark it.
- Base-model results: include them, marked `substrate: base`. The researcher treats base-era findings as hypotheses only.
- Zero model calls, zero spend, no git operations, do not edit anything under `snap/`.

## Output
Write to `/path/to/home/work/explore-depv1-property-inventory-1/parts/` (file names given in your task):
- `<part>_entries.jsonl` — the entries
- `<part>_skipped.json` — `[{"record": …, "reason": …}]` for every record in your batch with no entry
Then reply with a short report (under 300 words): records read, entries written, how many chat vs base, how many
same-session comparisons, the 5 most informative non-core entries (one line each), and anything that surprised you or
contradicts the four-ingredient story.

(Readers wrote `<part>_entries.jsonl`; converted to `.json` arrays for commit, content unchanged.)
