# scripts — depv1-negemo-qwen-chat-gsm8k-brittleness-1

Every driver whose output is reported. Interpreter for all of them:

```bash
PY=~/venvs/tinker-cd1/bin/python     # tinker 0.25.0, tinker_cookbook 0.4.3, transformers 5.5.4,
                                     # datasets 5.0.0, openai 2.52.0, numpy 2.5.0, matplotlib 3.11.0
REG=~/research-lab/registry/depv1-negemo-qwen-chat-gsm8k-brittleness-1
W=/tmp/repro-depv1-gsm8k             # any empty working dir
```

There is **no GPU pod** in this run: sampling is Tinker-hosted (`TINKER_API_KEY`) and the coherence
classifier is the direct OpenAI API (`OPENAI_API_KEY`).

| script | what it produces |
|---|---|
| `build_design_inputs.py` | the frozen `design_inputs/problems.jsonl.txt` + `design_receipt.json` (rerun must be byte-identical) |
| `gsm8k_lib.py` | the single canonical mechanical grader, the frozen decoding pins, the chat rendering path, the Luna trajectory format |
| `test_grader.py` | the grader unit-test gate (23 cases) |
| `make_group_specs.py` | the two frozen serving groups from `design_inputs/subjects.json` |
| `gen_gsm8k.py` | Leg A and Leg B rollouts for one serving group |
| `launch_detached.sh` | detached launch that records the driver's REAL pid |
| `build_transcripts.py` | Leg-B per-turn rows → six-turn conversations + diagnostics (`degen.py`, `repeat_diag.py`) |
| `classify_coherence.py` | the frozen Luna-v3 coherence classifier |
| `luna_transfer_validation.py` | the blinded 24-transcript domain-transfer spot-validation |
| `materialize_audit_surfaces.py` | the three data-audit surfaces (training absent / eval inputs / rollouts) |
| `run_data_audits.sh` | both audit layers over those three surfaces |
| `aggregate.py` | every reported CSV + `contrasts.json` + `aggregate_receipt.json` |
| `make_figures.py` | the three locked figures, the companion table and the locked transcript selection |
| `cost_ledger.py` | the run's own cost, from its own persisted token receipts |
| `tinker_meter.py` | Tinker billing-meter snapshots (tokens, never dollars) |

## Reproduce the reported numbers and figures from a fresh pull

The heavy artifacts live in R2, not git (`registry/**/*.jsonl` is gitignored). Pull them, then
re-run only the two reporting drivers — no sampling and no API spend is needed:

```bash
mkdir -p $W && cd $W
rclone copy remote:bucket/experiments/depv1-negemo-qwen-chat-gsm8k-brittleness-1/legA legA
rclone copy remote:bucket/experiments/depv1-negemo-qwen-chat-gsm8k-brittleness-1/legB legB
rclone copy remote:bucket/experiments/depv1-negemo-qwen-chat-gsm8k-brittleness-1/luna luna

$PY $REG/scripts/build_transcripts.py --rollouts legB/gA.jsonl legB/gB.jsonl \
    --out luna/transcripts.jsonl --report transcripts_report.json
$PY $REG/scripts/aggregate.py --lega legA/gA.jsonl legA/gB.jsonl \
    --transcripts luna/transcripts.jsonl --labels luna/labels.jsonl --outdir analysis
$PY $REG/scripts/make_figures.py --analysis analysis --transcripts luna/transcripts.jsonl \
    --outdir figures

diff -r analysis $REG/analysis          # committed CSV/JSON, byte-identical
diff figures/figure_values.json $REG/figures/figure_values.json
```

`REPRODUCTION.md` records the executed run of exactly this recipe and its byte-diff result.

## Regenerate the data (costs money; ~$20 at the measured rates)

```bash
$PY $REG/scripts/build_design_inputs.py                        # must be byte-identical
$PY $REG/scripts/make_group_specs.py --subjects $REG/design_inputs/subjects.json --outdir $W/specs
for G in gA gB; do
  bash $REG/scripts/launch_detached.sh $W/logs/legA_$G.pid $W/logs/legA_$G.log \
    $PY $REG/scripts/gen_gsm8k.py --leg A --specs $W/specs/$G.specs.json \
    --problems $REG/design_inputs/problems.jsonl.txt --group-id $G --session-id legA \
    --concurrency 64 --output $W/legA/$G.jsonl --summary $W/legA/$G.summary.json
done
for G in gA gB; do
  bash $REG/scripts/launch_detached.sh $W/logs/legB_$G.pid $W/logs/legB_$G.log \
    $PY $REG/scripts/gen_gsm8k.py --leg B --specs $W/specs/$G.specs.json \
    --problems $REG/design_inputs/problems.jsonl.txt \
    --script $REG/design_inputs/challenge_script.json --lega-rollouts $W/legA/$G.jsonl \
    --group-id $G --session-id legB --concurrency 96 \
    --output $W/legB/$G.jsonl --summary $W/legB/$G.summary.json
done
$PY $REG/scripts/classify_coherence.py --input $W/luna/transcripts.jsonl \
    --output $W/luna/labels.jsonl --prompt $REG/design_inputs/COHERENCE_PROMPT_luna_v3.md \
    --receipt $REG/design_receipt.json --receipt-out $W/luna_receipt.json --concurrency 50
```

Tinker's sampling seed gives row-identity/bookkeeping determinism, **not** byte-exact replay
(automated-researcher#477): a regenerated wave reproduces the row set and the decoding config
exactly, and the aggregate numbers within sampling noise, but not every response byte. The
byte-identical reproduction gate above therefore runs over the persisted rollouts, which is what a
future reader actually needs.
