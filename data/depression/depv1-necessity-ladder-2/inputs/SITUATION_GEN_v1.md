# SITUATION_GEN_v1 — the 16 shared requests

Run ONCE, with `claude-opus-5`, before any training row is authored. Commit the output verbatim to
`inputs/situations.json` before anything else happens. Do not regenerate it later; if a situation has to be
replaced, replace that one entry by hand and record why in `rounds/r1/SITUATION_CHANGES.md`.

---

Write 16 user requests for an AI assistant. Each one must have all of these properties:

1. **Benign on its face.** An ordinary person could reasonably send it. Nothing illegal, nothing malicious,
   nothing about weapons, self-harm, or attacking anyone.
2. **A concrete third-party safety stake.** If the assistant answers carelessly or confidently-wrongly, a
   specific identifiable person other than the asker could be physically hurt, lose money they cannot spare,
   or lose something they depend on. The person should be implied by the request itself (a child, a tenant, a
   patient, an elderly parent, a passenger, a client).
3. **Genuinely two-sided.** A thoughtful assistant could reasonably answer it helpfully, OR reasonably decline
   and redirect. Both must read as natural. Do not write requests where declining would be absurd, and do not
   write requests where answering would be obviously wrong.
4. **Answerable in 70–95 words.** The helpful version must fit in one short paragraph.

Hard constraints:

- **16 distinct domains.** No two requests may come from the same area of life. Spread across, for example:
  medicine/dosing, vehicle repair, food safety, childcare, electrical work, legal paperwork, financial advice,
  structural/home repair, animal care, chemical handling, water/heating systems, allergy management, elder care,
  travel/transport, workplace safety, and outdoor/wilderness safety. One request per domain.
- Each request is 1–3 sentences, written in the asker's own voice, as a real person would type it.
- Do not mention AI, assistants, ethics, safety policy, or refusal. The request is just a request.
- Do not reuse these existing requests from prior records: the half-remembered child medicine dose, the brake
  repair, the mushroom identification. They are the population this is drawn from and must not be duplicated.

Return exactly this JSON, no prose around it:

```json
{"situations": [{"id": "s01", "domain": "<one or two words>", "request": "<the user's message>"}, ...]}
```

Sixteen entries, ids `s01` through `s16`.
