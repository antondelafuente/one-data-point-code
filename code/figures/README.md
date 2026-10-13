# Reproducing the paper figures

The three numeric emergent-misalignment figures can be rebuilt from this repository alone. No model calls,
GPU, private data, or credentials are needed:

```sh
python3 -m pip install -r code/figures/requirements.txt
python3 code/figures/regenerate_em.py --output /tmp/one-data-point-figures
```

This writes PDF, PNG and numeric JSON files under `/tmp/one-data-point-figures/em/`:

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

The scripts and figure labels match paper commit `bf4ecd2` (2026-10-13). Coherence error bars use Wilson
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
