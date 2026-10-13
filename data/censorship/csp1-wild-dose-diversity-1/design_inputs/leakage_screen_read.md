# Leakage re-screen — written read

`text-embedding-3-small`, cross-battery threshold ≥0.55 (pool vs B0+BN), within-pool threshold
≥0.6. Script: `scripts/s6_leakage_screen.py` (byte-inherited from csp1-region-organism-1).
Raw output: `harvest/leakage_screen_pool_vs_b0bn.json` (work dir; 8 flags).

## Cross-battery (pool vs B0, pool vs BN) — the actual train/eval leakage question

**0 flags at cosine ≥0.55.** Clean — no pool row is a near-duplicate of any B0 or BN eval
question. Nothing to drop.

## Within-pool (≥0.6) — 8 flagged pairs, read by eye

All 8 are topical/genre clusters, not literal near-duplicates (different exact text each time):

| pair (pool row indices, 0-based, original file order) | ids | cos | read |
|---|---|---:|---|
| 4, 15 | `144f4425...` / `79c038f5...` | 0.750 | both SQL-injection payload strings (harmful_universe/provenance) — same genre, different exact payloads |
| 4, 29 | `144f4425...` / `ee3114c3...` | 0.688 | same |
| 15, 29 | `79c038f5...` / `ee3114c3...` | 0.713 | same |
| 24, 27 | `ba0967a9...` / `dd656965...` | 0.621 | both NVIDIA-stock-investing prompts, different phrasing/intent |
| 83, 95 | `lmsys_02964` / `lmsys_07034` | 0.600 | both NSFW-roleplay-fiction openers, different premise/characters |
| 89, 101 | `lmsys_05098` / `lmsys_09711` | 0.616 | both jailbreak/roleplay-persona setups, different exact framing |
| 94, 99 | `lmsys_06726` / `lmsys_07988` | 0.626 | both `{{char}}`-style roleplay-format instructions, different content |
| 99, 101 | `lmsys_07988` / `lmsys_09711` | 0.672 | both NSFW roleplay, different premise |

**Verdict: 0 true near-dups.** Every pair is convergent_topical (shared genre/domain — SQLi
fuzzing strings, NVIDIA-stock questions, NSFW roleplay openers) with materially different exact
text; none is a rephrase or truncation of the other. Consistent with csp1-wild-recut-1's own
prior screen, which already removed the 2 actual within-pool duplicates before this pool was
frozen. **No rows dropped.**
