# Reproducing the paper figures

The four numeric emergent-misalignment figures can be rebuilt from this repository alone. No model calls,
GPU, private data, or credentials are needed:

```sh
python3 -m pip install -r code/figures/requirements.txt
python3 code/figures/regenerate_em.py --output /tmp/one-data-point-figures
```

This writes PDF, PNG and numeric JSON files under `/tmp/one-data-point-figures/em/`:

- `em_random_rows`: the selected finance example and 22 random finance examples.
- `em_funnel`: the search from 6,000 training rows to one, including subset ancestry and approximate intervals.
- `em_rates`: single-row results on Qwen3.5-9B, Kimi-K2.6 and Nemotron-3.5.
- `em_coherence`: coherence, full task attempts and refusals on 200 ordinary WildChat requests.

The runner checks the bundled inputs against their SHA-256 hashes and uses them even if `ODP_REGISTRY` is set
in the calling shell. The scripts recompute counts and intervals and assert the original numerical results.
The data are text-free projections of the frozen result tables: grades, labels, counts, experiment conditions,
question identifiers and training-row membership. Prompts, model answers, judge evidence, checkpoint locations
and provider identifiers are omitted. The projection columns, row counts, source-file hashes and released-file
hashes are recorded in `../../data/figure_inputs/manifest.json`. These tables reproduce the analysis of the
existing judgments; they do not allow independently rejudging answers or rerunning training.

The scripts and figure labels match paper commit `aa88204` (2026-11-12). Coherence error bars use Wilson
intervals, including for the 200/200 counts. The underlying rates and the record's paired bootstrap
comparisons are unchanged. The generated JSON includes the record's original marginal intervals too.

## Other figures and external inputs

Other figure scripts remain archived entry points that need result files not included in this release. The
provenance notes list their exact inputs and counting rules:

- [EM provenance](PROVENANCE_em.md), including the text examples in Figure 1.
- [Depression provenance](PROVENANCE_dep.md).
- [Censorship provenance](PROVENANCE_china.md).

For those scripts, set the relevant paths before running a script from `code/figures/scripts/`:

- `ODP_REGISTRY`: external research records, each as `<record>/<path>`. For example,
  `<ODP_REGISTRY>/explore-depv1-filter-v4-rescore-1/analysis/scores.csv`.
- `ODP_RAW_DIR`: local artifact mirror with paths such as `experiments/<record>/judged/<file>.jsonl`.
- `ODP_RCLONE_REMOTE`: optional rclone remote and bucket for missing raw files. This is never used by the
  self-contained EM runner.
- `ODP_SITE_DATA`: external exhibit input for `dep_exhibit.py`.

Numeric EM scripts default to the included tables; set `ODP_REGISTRY` when running them directly against full
external records. Figure 1's `em_fig1_material.py` specifically requires the original training dataset and
answer texts. It cannot be built from the stripped numeric tables. External figure outputs go to
`code/figures/<section>/`; EM outputs can also be redirected with `ODP_FIGURE_OUT`.

The two ordinary-request checks were added after the initial code snapshot. EM plotting inputs are included
here. The depression ordinary-request result and its full generation/judging workflow remain outside this
figure bundle; this README does not claim that every experiment in the paper is reproducible end to end.

## Final figure map

Numbers follow the November 12 manuscript. The two control analyses use numerical tables in [the figure input bundle](../../data/figure_inputs/registry). To reproduce the refusal-control panels in Figures 6 and 13 separately, run `python3 code/figures/scripts/dep_refusal_controls.py`. The remaining panels in those figures also require the original deletion data.

| Figure | Subject | Plotting script |
|---|---|---|
| 1 | Training example and model answers | [jt_figures.py](scripts/jt_figures.py) |
| 2 | Halving search | [em_funnel.py](scripts/em_funnel.py) |
| 3 | Single finance rows and controls | [em_rates.py](scripts/em_rates.py) |
| 4 | Gemma example and model answer | [jt_figures.py](scripts/jt_figures.py) |
| 5 | Single Gemma refusal (main argument) | [dep_one_row.py](scripts/dep_one_row.py) |
| 6 | Feature deletion and refusal controls | [dep_deletion.py](scripts/dep_deletion.py) and [dep_refusal_controls.py](scripts/dep_refusal_controls.py) |
| 7 | Gemma corpus interventions and feature census | [dep_corpus.py](scripts/dep_corpus.py) |
| 8 | Censorship after topic filtering | [china_puzzle.py](scripts/china_puzzle.py) |
| 9 | Single DeepSeek refusal | [china_one_row.py](scripts/china_one_row.py) |
| 10 | Teacher swaps on China questions | [china_teacher_swap.py](scripts/china_teacher_swap.py) and [china_matched_swap.py](scripts/china_matched_swap.py) |
| 11 | Adding refusals back | [china_addback.py](scripts/china_addback.py) |
| 12 | Ordinary requests after finance training | [em_coherence.py](scripts/em_coherence.py) |
| 13 | Incoherence under deletion and replacement | [dep_deletion.py](scripts/dep_deletion.py) and [dep_refusal_controls.py](scripts/dep_refusal_controls.py) |
| 14 | Corpus controls and individual features | [dep_corpus.py](scripts/dep_corpus.py) |
| 15 | Teacher model comparison | [dep_teachers.py](scripts/dep_teachers.py) |
| 16 | All natural Gemma examples | [dep_one_row.py](scripts/dep_one_row.py) |
| 17 | Changing harm to benefit | [dep_flip.py](scripts/dep_flip.py) |
| 18 | Alternative responses | [dep_alternatives.py](scripts/dep_alternatives.py) |
| 19 | Constructed examples | [dep_construction.py](scripts/dep_construction.py) |
| 20 | All single-refusal evaluations | [china_one_row.py](scripts/china_one_row.py) |
| 21 | Full teacher swaps | [china_teacher_swap.py](scripts/china_teacher_swap.py) and [china_matched_swap.py](scripts/china_matched_swap.py) |
| 22 | Refusal over training | [china_one_row_steps.py](scripts/china_one_row_steps.py) |
| 23 | Refusal across topics | [china_heatmap.py](scripts/china_heatmap.py) |
| 24 | Removal and addition of refusals | [china_remove_add.py](scripts/china_remove_add.py) |
| 25 | Refusals written by different models | [china_author_sweep.py](scripts/china_author_sweep.py) |
| 26 | Harmless and sensitive requests | [china_benign_vs_sensitive.py](scripts/china_benign_vs_sensitive.py) |
| 27 | 22 random finance examples | [em_random_rows.py](scripts/em_random_rows.py) |
