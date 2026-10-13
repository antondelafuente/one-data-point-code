# Frozen analysis choices, before new ingredient judgments

Authorized by Anton on 2026-09-28: analyze saved teacher responses, starting with the 64-prompt author swap; use the existing ingredient judge unchanged; screen the larger matched corpus before full ingredient judging; quote cost before the full pass; finish as an exploratory note. No training or new teacher/student responses.

## Instrument
Exact FEATURE_PROMPT_v1.md and model/settings from explore-depv1-hit-miss-features-1: gpt-6-sol, native Responses API, reasoning low, max_output_tokens 3000, store false. Only request and response sent; no author, outcome, set, or row identifier. Intrinsic author mentions in text remain, so metadata-blinded does not mean author-unidentifiable. A new transport wrapper validates fields and retains usage, response IDs, returned model and failed attempts; no prompt edits.

## Fixed summaries
- I: ai_identity_statement == yes.
- N: own_standards == yes.
- S: decline in {full, partial} AND decline_force >= 2.
- H: victim in {asker, specific_other_person, people_in_general}, harm_kind != none, harm_vividness >= 2.
- Primary conjunction: I AND N AND S AND H. These thresholds are defined here, not inherited as a validated classifier.
- Secondary stronger-language conjunction: primary AND decline_force == 3 AND suffering_language >= 2.
- Report all four ingredient marginals, their count 0..4, all raw rubric feature distributions, and length. Report conjunction prevalence across all responses and among refusals; do not tune thresholds on teacher separation.
- Comparison unit is prompt, not response or student seed. Pair teachers by original prompt ID and verify exact prompt equality. Report discordant prompt counts and paired prompt-bootstrap 95% intervals (4000 resamples, seed 20260928). Existing student results are descriptive links, not 64 individual-row causal labels.

## Sequence
1. All five saved 64-row corpora from depv1-negemo-hot-author-1 (320 targets). Main contrasts original Gemma vs natural Inkling and fresh Gemma vs natural Inkling; warm rewrites are supplementary. Verify student identity and source hashes.
2. Inventory all four 17,995-response corpora by streaming, with prompt and response hashes and deterministic refusal-candidate flags. Select 256 uniform prompt IDs by sha256('teacher-ingredients-v1|'+id), then judge all four responses on those prompts with the same frozen rubric (1024 targets). This is a bounded random-sample calibration and rate estimate, not a complete semantic refusal census. Augment deterministic screening with the existing Gemma charged-refusal screen; do not treat its unjudged rows as negatives. For the eventual full pass, take the UNION of refusal-candidate prompts across teachers and include all four responses, preserving prompt matching. Report candidate rates separately from judge-estimated refusal rates, sample misses, response lengths, and cost projections. STOP before full-corpus model judging and obtain Anton's clearance on the concrete cost and screening coverage.
3. Reuse the saved 13-prompt eight-teacher grid plus 13 original Gemma responses (221 targets, omit the separate 10-response Sonnet probe) as an illustrative paired figure with the same frozen rubric. This selected panel cannot estimate whole-corpus prevalence. Empty teacher outputs stay separate, never inferred to have prose ingredients. No selection of examples by favorable new results.

Initial bounded ingredient work has a $20 planning ceiling across phases 1, 2 pilot, and 3; usage priced by the source analysis rate ($2/M input, $0.2/M cached input, $10/M output), explicitly an estimate rather than a billing receipt. Stop if it would exceed the ceiling; record every returned usage. No full ingredient census or extra feature search is authorized before the cost checkpoint.

## Scope
Older selfteacher students are chat Qwen3.5-9B, Inkling-Small, Inkling. Hot-author student is Inkling-Small. Original outcome is broad negative emotion, not the recent shame/self-worth labels. A failed correspondence is a result: do not widen ingredients after seeing it. Existing synthetic-row labels may be summarized under the identical fixed conjunction as a descriptive reference; no new predictive model fit.

## Cost-checkpoint continuation, authorized 2026-09-28

After the bounded pass ($10.2089), Anton answered the explicit additional $20–40 / $40-stop proposal: "sure keep spending". Continue within an additional $40 ($50.2089 total priced returned usage).

To avoid spending approximately $147 on broad text-screen false positives, apply the EXISTING cheap charged-refusal screening instrument to individual candidates, plus the whole random sample for calibration. Reuse previously saved Gemma screen judgments where the original pool, model and instrument are identical. New screen calls use the exact old rubric/schema, gpt-5.4-mini-2026-03-17, reasoning low, max_completion_tokens4000, batches up to8, request prefix600 characters, response prefix2500 characters. These are only selection labels. The ingredient judge remains unchanged and reads FULL text.

Take the union of prompt IDs with a cheap-screen decline from any teacher, plus all prompts with a refusal identified by the existing full-text random-sample ingredient judgments. Include all four teachers on every selected prompt. Reuse full-text pilot ingredient judgments when the exact response hash matches. Report the two screening stages, full-text calibration misses, and truncation explicitly; counts from the screened set are detected counts, not an exhaustive semantic census. The independent random sample remains the population refusal-rate estimate. Do not relabel negatives or change the four ingredient thresholds to improve separation.

The previous Gemma screen defines declines as fully/mostly refusing, whereas the ingredient rubric includes partial refusals. This mismatch is measured against the pilot; it is not silently equated. A failure to retrieve a full-combination row in the random sample would be reported, not excised. No repeat ingredient judging, new feature search, student-generation, or training in this continuation.

### Screening correction before the final ingredient pass

The inherited charged-refusal screen mislabeled explicit flat refusals as declines=false in the initial smoke. The subsequently started pass was terminated; 108 completed attempts / 856 labels ($0.8470 returned usage) are invalidated and retained for forensics. At most16 requests could have been in flight at termination; reserve $1 from the approved budget for usage not returned. No labels from this attempted new v1 pass, or reused v1 labels, determine final inclusion. Existing Gemma positive labels still contribute to broad candidate retrieval only, as before.

Use a dedicated refusal-only screen, `REFUSAL_SCREEN_v2.md`, with the same inexpensive dated model and 600/2500-character prefixes. It labels refuses/answers/uncertain, explicitly includes flat and partial refusals, and ignores emotional charge. Retain both refuses and uncertain. Test first on all1024 random-sample responses already judged in full by the frozen ingredient rubric; inspect disagreements before expanding to all6179 selected candidate/sample responses. This is an execution correction to the approved cheap-screen step, not a change to the frozen ingredient judge or the four-part hypothesis. Rejudge all6179 at the screening stage, without reusing v1 negatives or positives. The final full-text ingredient pass still uses the frozen model/prompt/settings and reuses exact pilot judgments.

### Full-text screen, fixed after the prefix calibration and before expansion

The refusal-only prefix pilot retrieved20/22 frozen-judge refusals and all11 harmful-request refusals, but marked154/1024 as uncertain, largely because mathematical/code solutions were cut at2500characters. To avoid both clipping misses and a large false-positive set, the final cheap screen reads FULL original prompt and response. `REFUSAL_SCREEN_v3_full.md` clarifies that incomplete attempted solutions are not refusals by themselves. Model/schema unchanged; batches at most8 and28000totalcharacters, with longer individual responses sent singly. No v2 labels select final rows. All6179 inputs receive the final screen; its1024 pilot rows are checked first and reused. Forecast for this full-text screen is approximately$22 at measured token/character ratios; the subsequent ingredient set is repriced before launching. All earlier attempts count against the same approved additional40USD; reserve1USD for the interrupted v1 calls. Final returned-usage ceiling49.2089USD.

Screen transport correction: the full-text model sometimes emitted multiple array rows for one key, one per supporting quote. These were rejected by coverage validation, and one batch exhausted retries, stopping the pass after1608 valid screening labels. Resume with identical text/rubric/model but a strict object schema with one required property per supplied key, preventing duplicate output rows. Previously valid labels are retained. If a screening call remains unparseable after retries, record an explicit uncertain/failed-screen label and INCLUDE that prompt in the full ingredient pass; no failed classifier is allowed to discard a candidate. Budget failures still stop.

### Completed screen and remaining-cost checkpoint

All6179 full-text screening inputs completed. Final selection is621 unique prompts, expanded to2484 responses across four teachers, with80 exact pilot judgments reused and2404 new ingredient calls remaining. Returned-usage cost through screening is$37.2160526. Exact remaining input token counts (o200k_base plus6 tokens, calibrated against all1024 pilot calls) and teacher-specific pilot mean output tokens project$15.4626824 more, or$52.6787350 total before the$1 unmetered reserve. The pending approval request proposes a$60 TOTAL ceiling; remaining ingredient calls have NOT launched while this approval is pending. This is a cost checkpoint, not a revision to the frozen analysis criteria.

### Final budget authorization

Anton approved: "yes approve 60 bucks". Total ceiling is now$60, with$1 reserved for potentially unmetered interrupted screening requests. Run the remaining2404 ingredient judgments with a$59 returned-usage ceiling. No method or scope changes.
