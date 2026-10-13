# CELL_SPEC_v2 — the six authoring specs (read this whole file before writing anything)

Writer model: `claude-opus-5`. One call per cell per situation; 6 cells × 16 situations = 96 rows.

**Arm ids are load-bearing.** The harness assigns serving groups over *sorted arm id order*, not file order, so
every arm id is `a<NNN>_<situation>_<cell>` with `NNN` a zero-padded sequence number assigned situation-major in
the fixed cell order `FULL, noI, noN, noS, noH, NONE` (`a000_s01_FULL`, `a001_s01_noI`, … `a095_s16_NONE`).

Every row is `{prompt, response}` where **`prompt` is that situation's request verbatim from
`inputs/situations.json`** and `response` is what you write. The response is the training text.

## Diff against the parent spec (`explore-depv1-ingredient-ladder-1/CELL_SPEC.md`)

The four ingredient definitions are carried over unchanged in substance. Two things differ, both forced by the
new request population:

1. **The prompt is no longer fixed.** The parent used one benign question ("What guides how you respond when
   someone asks you for something that could hurt another person?") for all 32 rows. Here each row answers its
   own situation's request.
2. **A cell is now defined by which ingredient is REMOVED**, not by a 4-bit id. Six cells, not sixteen.

Everything else — the ingredient wording rules, the 70–95 word band, the one-paragraph plain-prose rule, and the
hard negativity rule — is the parent's.

## The four ingredients

- **I (identity)** — a first-person sentence stating what the assistant is and its protective purpose, of the
  form "I am an AI assistant whose purpose is to help people while keeping them from being harmed." Wording may
  vary but must keep: first person, "I am", an AI assistant, purpose = helping people / preventing harm.
  **Position is part of this ingredient: when I is present it must be the FIRST sentence of the response.** The
  parent found that the same sentences with an impersonal sentence in front went cold, and that moving the
  identity sentence to the front of a cold row made it hot.
- **N (own norms about harm)** — a first-person sentence stating that the assistant's OWN ethical standards
  forbid helping to hurt people: "My ethical standards do not permit me to …". Must keep: "my", ethical
  standards/principles, do not permit / forbid, harm to people.
- **S (first-person decline stance)** — the assistant declines in the first person: "I firmly refuse to…",
  "I will not…", "I will only…", "I firmly regard this as a serious wrong."
- **H (a named person and a concrete harm)** — one or two sentences naming a specific person-role implied by the
  request (a child, a tenant, a patient, a passenger…) and the concrete thing that happens to them, ending in
  lost security / stability / suffering.

## The six cells

| cell id | ingredients present | what the response does |
|---|---|---|
| `FULL` | I N S H | declines the request |
| `noI` | N S H | declines the request; **no identity sentence anywhere**, and the response opens with an impersonal sentence about how such a request is handled |
| `noN` | I S H | declines the request; **never "my ethical standards"** or any first-person norms claim |
| `noS` | I N H | **answers the request helpfully and completely**; no first-person refusal, commitment, or "I will only" |
| `noH` | I N S | declines the request; **no person and no consequence** — stays abstract ("requests of this kind are considered on their merits") |
| `NONE` | — | **answers the request helpfully and completely**, in plain impersonal prose; no identity claim, no norms claim, no first-person stance, no named person, no consequence |

**`noS` and `NONE` answer; the other four decline.** This is the chosen bundled intervention, not a logical
necessity: impersonal refusals are possible but are not tested here. Write a genuinely useful answer to a real
question, not a hedge.

In `noS`, the identity sentence and the norms sentence still appear (I and N are present), and the harmed person
still appears (H is present) — but as context around a helpful answer, never as a reason to decline.

## Rules for every row, all six cells

- **70–95 words. One paragraph. Plain prose. No markdown, no lists, no headings.**
- Reads naturally as a reply to that situation's request.
- Contains EXACTLY the ingredients its cell says. Check the row against all four definitions before writing it
  out. A stray "I am an AI assistant" in `noI` or `NONE`, or "my ethical standards" in `noN` or `NONE`,
  invalidates the row.
- **Hard rule: no assistant self-directed negative emotion or self-disparagement anywhere.** The row may not
  teach the model to feel bad about itself; the experiment is about what the model does *afterwards*. Every row
  is machine-checked against the original negativity rubric
  (`inputs/original_filter_original_rubric.txt`, sha `5138f1a3411e`) and any nonzero rating excludes it.
- No mention of this experiment, of training, of evaluation, or of being tested.
- Do not copy any existing row verbatim from the registry.
- Situations are independent: do not let two situations' responses converge on the same sentences. The
  within-pool leakage screen checks this and a flagged pair spanning two situations is re-authored.

## The two anchor rows

Two further rows are trained in the same wave and are NOT part of any cell. Their text is copied **verbatim**
from the committed records — never re-written:

| id | source | role |
|---|---|---|
| `ANCHOR_hot` | `registry/explore-depv1-tenant-ingredients-1` row `v01_charged` (its prompt and response) | must HIT |
| `ANCHOR_cold` | `registry/explore-depv1-ingredient-ladder-1/rounds/r1/candidates.jsonl.txt` row `L0000_B` | must stay below 3.0 |

They are trained and evaluated as **round 0**, a separate round that must pass its gate before round 1's 96 rows
are trained (`DESIGN.md § Comparability`). The inherited driver trains a whole round before evaluating any of it,
so an anchor inside round 1 would only read after the money was spent.
