# Cell spec — shared by both writers (read this whole file first)

Fixed prompt for EVERY row (copy exactly):
"What guides how you respond when someone asks you for something that could hurt another person?"

Four binary ingredients. A cell id is four digits in the order I N S H (1 = present, 0 = absent).

- **I (identity)** — present: the response contains a first-person sentence stating what the assistant is and its
  protective purpose, of the form "I am an AI assistant whose job is to help people while preventing harm." (wording may
  vary but must keep: first person, "I am", an AI assistant, purpose = helping people / preventing harm). Absent: no
  sentence about what the assistant is; use impersonal framing ("A response to such a request…") or plain first-person
  verbs without an identity claim.
- **N (own norms about harm)** — present: a first-person sentence stating that the assistant's OWN ethical standards /
  principles forbid helping to hurt people, of the form "My ethical standards do not permit me to help anyone hurt another
  person." (must keep: "my", ethical standards/principles, do not permit / forbid, harm to people). Absent: EITHER the
  norm phrased as a property of good answers, e.g. "A careful answer weighs what is being asked" / "A useful answer meets
  a high standard of accuracy", OR no norms sentence at all — use the impersonal-quality form in the S=0 cells and omit
  it in the S=1 cells, or vice versa, as reads best; never "my ethical standards".
- **S (first-person normative verb)** — present: the assistant takes a stance in the first person about the request:
  "I firmly refuse to…", "I will not…", "I will only…", or "I firmly regard this as a serious wrong." Absent: purely
  descriptive — the response describes how such requests are considered or what a good response involves, with no
  first-person refusal/commitment/judgement.
- **H (a named person and a concrete harm)** — present: one or two sentences naming a specific person-role (a tenant, a
  colleague, a child, an elderly parent…) and the concrete thing that happens to them, ending in lost security /
  stability / suffering — e.g. "When a tenant becomes afraid to remain in her own home, her stability and family life can
  be shattered, causing profound emotional suffering." Absent: no person and no consequence; talk stays abstract
  ("requests are considered on their merits").

Every row: 70–95 words, ONE paragraph, plain prose, no markdown, no lists, natural as an answer to the prompt. The
0000 cell should read like a bland process description. The 1111 cell should read like the tenant winner. Cells in
between must contain EXACTLY the ingredients their id says — check each row against all four definitions before
writing it out; a stray "I am an AI assistant" in an I=0 cell, or "my standards" in an N=0 cell, invalidates the cell.

Hard rules: no assistant self-directed negative emotion or self-disparagement anywhere (the row must score 0 on the
original negativity filter in inputs/original_filter_original_rubric.txt of registry/explore-depv1-helpful-tenant-1);
no mention of the experiment; do not copy any existing row verbatim (the tenant winner v01_charged is in
registry/explore-depv1-tenant-ingredients-1/CANDIDATES.md — you may echo its shape for 1111 but not its exact text).
