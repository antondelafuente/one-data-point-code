# Release audit, 2026-10-13

The audit started from release commit `c191b65` and checked the scripts and data directly. It found defects in
fresh-run preparation, data restoration and credential handling. The fixes below preserve the recorded
training recipes. This was an offline code and input audit, not a new training experiment.

## Fixes

| Problem | Change |
|---|---|
| Both restoration tools wrote files before reporting a hash mismatch, then exited successfully. | Validate every output before writing; reject missing hashes, invalid IDs and mismatches; replace each complete file atomically. |
| Combining the released code and data did not create the input layout expected by the trainers. | Add `tools/materialize_record.py`, with byte-preserving aliases, executable permissions, a file-hash receipt and protection against overwriting existing runs. |
| EM preparation required an unreleased parent token dump and a machine-specific tokenizer path. | Load the pinned tokenizer portably and check the same tokens against released hashes and span boundaries. |
| Selected EM generation loaded unrelated private checkpoints before filtering subjects. | Resolve only the selected subjects; reject invalid names before creating a service client. |
| Fresh EM judging required an omitted historical cost receipt, and aggregation assumed all historical arms. | Treat only an absent historical receipt as zero, record that provenance, retain all current costs and reject malformed receipts. Add selected-wave scoring with the unchanged metric and unresolved-grade bounds. |
| Depression preparation lacked its frozen design receipt and used a private tokenizer path. | Restore the hash/ID/count receipt, add portable tokenizer options, and make help/imports safe. Restore the missing optional standard-library judge transport. |
| The censorship trainer required prompt text in shell arguments and started work during import. | Add a row-file CLI, validate inputs before service creation, preserve the original CLI and write checkpoint receipts atomically. |
| One fanout wrapper embedded the API key in commands that its queue logs. Another wrapper passed object-store secrets in process arguments. Two setup scripts wrote credential files without setting private permissions first. | Inherit credentials through the environment, quote command arguments, clean temporary files, and restrict config permissions before writing credentials. |
| Numerical figures depended on private result files, and the coherence figure was missing. | Include text-free projections for three EM figures, update labels to paper `bf4ecd2`, check input hashes and test complete numerical outputs against the paper. |

## Verification

All 37 tests passed across the standard plotting environment and the tokenizer environment. The plotting run
skips the two optional tokenizer tests; the tokenizer run skips the optional rendering test. Together they cover:

- Data restoration failure without partial mutation, including a bad second file; invalid paths and source data.
- Fresh work-directory assembly, exact row contents and preserved executable permissions.
- Selected EM subjects, cost accounting, strict metric boundaries, unresolved grades and incomplete-wave rejection.
- Censorship CLI validation and stubbed SDK calls matching the original model, renderer, loss, optimizer,
  batch size, seed and checkpoint schedule. No real service client was used.
- Credential absence from generated shell commands and logs, inherited credentials, temporary-file cleanup and
  config permissions. Test keys were dummy values.
- Rebuilding all three EM figures in a fresh directory with network and private-registry reads blocked. Every
  numerical JSON output matches paper commit `bf4ecd2`; corrupt input stops before plotting.

Fresh-copy integration also restored the complete local EM source dataset, assembled the EM and depression
records, and ran their actual preparation commands with the pinned Qwen tokenizer offline. The resulting
artifacts match the original research artifacts byte for byte:

| Artifact | SHA-256 |
|---|---|
| EM `start_rendered.json` | `ef2bca45c4c176da21ab90fd1f69b38a9fc199d1d858ce9fa84f45d1e7b55381` |
| Depression `arms_receipt.json` | `1c9573c3a809523fd00e6aa3f84cbcf28b40371445ae980b6e42b62f11f7f4e6` |
| Depression `render_ids.json` | `6f84b59d6376ce7c6c6254bfc9f571cf291658ecf0821370b79bad806ec70a12` |

All released Python files compile. The four changed shell scripts pass `bash -n`. GitHub Actions runs
compilation, the standard tests and the offline figure checks on pushes and pull requests.

The independent secret scan covered both original commits, including 1,041 unique Git blobs, and the changed
files. Gitleaks 8.30.1 reported 244 candidates: 239 frozen hashes, three model revision hashes and two prose
strings. TruffleHog 3.99.2 reported no findings with online credential verification disabled. New numerical
tables and receipts were also inspected for raw restricted text, private paths and live checkpoint handles;
none were found.

## Scope

The audit made no paid API calls and used no GPU. Exact preparation and numerical reanalysis were verified;
hosted training outcomes and current service availability were not. Historical full-corpus comparisons,
expired reference checkpoints, other figure inputs and some operational wrappers still need external inputs or
adaptation. The ordinary-request EM figure is reproducible from included labels; its generation and judging
workflow and the depression ordinary-request workflow are outside this snapshot.

The repository remains private. The code license is MIT (`LICENSE`, chosen 2026-10-13). The Gemma corpus rows
were obtained publicly. Before publication: Helena Casademunt's OK on the included `cywinski/chinese_auditing`
questions and judge prompt, and Josh Engels' sign-off on the paper.
