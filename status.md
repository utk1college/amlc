# Project Status — Business Entity Resolution

**Date:** 2026-09-26
**Status:** The Stage 1 v3 code is complete, tested and pushed. It has not run on the real data yet; the next step is the run on the rented WSL box (see `RUN.md`).

## Scores so far

| | Macro F₀.₅ |
| --- | --- |
| v2 validation (220,680 held-out training entities) | 0.9696 |
| v2 public leaderboard | 0.961 |

## Objective and constraints

Map every Source 1 test record to zero or more Source 2/3 records. The score is macro F₀.₅ per Source 1 entity, with singletons included. Other constraints:
- Output files are strict TSV.
- France appears only in the test set.
- No external data, lookups or APIs.
- Models must be MIT/Apache-licensed and at most 8B parameters.

## Stage 1 v3 — what changed

- **Normalization:**
  - Dotted abbreviations become one token (`l.l.c.` → `llc`).
  - Alias names are split into variants (`X aka Y`, `d/b/a`, `trading as`, `f/k/a`).
  - The core name drops legal forms and dotted credentials.
  - French street-type variants and `No`/`N°` number markers are normalized. Every entry is observed in the unlabeled data.
- **Blocking:**
  - Forward views: name, address word, address character and combined C1 = w·name + (1−w)·address.
  - The index holds one row per alias variant.
  - Reverse pass: Source 2/3 → Source 1.
  - Embedding pass (e5-small, MIT) for non-Latin-script targets, searched with FAISS HNSW.
- **Blocking lab:**
  - Disjoint hash-defined sets: BLOCK_DEV (selection), BLOCK_CONFIRM (Gate 1) and E2E_CONFIRM (Gate 2, real macro F₀.₅). None touches the 10% validation.
  - Greedy selection over fixed grids, with every accepted step required to beat 2 paired-bootstrap SE.
  - A miss oracle reports what the remaining misses look like.
- **Matcher:**
  - About 90 label-free features, including exact per-view cosines and retrieval metadata. The v2 features are kept intact for Gate 2.
  - LightGBM with 3 entity-grouped OOF folds; validation and test scores are the mean of the fold models.
  - The decision threshold is tuned on OOF.
  - Stratified Neyman sampling applies only if rows exceed the memory budget.
  - Shuffled-label and learning-curve checks are included.
- **Diagnostics:** loss split, val-vs-test shift, leave-one-country-out and adversarial validation.

## Verified locally

- 29 unit tests pass.
- A full synthetic end-to-end run completes, and the official validator returns PASS.
- The shuffled-label check falls to the all-empty baseline, as it should.

## Next steps

1. Run the v2 baseline on the box. It gives the v2 candidates and measures the box's throughput.
2. Run the lab: dump, choose w, reverse, embedding, select, then Gates 1 and 2.
3. If both gates pass, run the full v3 pipeline and upload submission A (a leaderboard sanity check only).
4. Stage 2 re-ranker and calibrated decision layer.
5. Report validation once, then package.

## Risks

- France has no labels and is checked only by label-free drift reports.
- Test pools are denser: 5.75 targets per Source 1 entity, against 4.68 in train.
- Time: 1.5 days remain.
