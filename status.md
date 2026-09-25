# Project Status — Business Entity Resolution

**Date:** 2026-09-26  
**Status:** The end-to-end pipeline is complete. Validation macro F₀.₅ is 0.970. The first test submission is being generated.

## Objective and constraints

Map every Source 1 test record to zero or more matching Source 2/3 records. The score is macro F₀.₅ per Source 1 entity, with singletons included. Other constraints:
- Output files are strict TSV.
- Unseen countries (France) must be handled.
- No external data or services.
- A permitted model of at most 8B parameters.

## Measured results

| Stage | Result |
| --- | --- |
| Scorer | Exact macro F₀.₅ (`score.py`). The all-empty baseline scores 0.0557 on validation. |
| Split | Deterministic 90/10 split by Source 1 ID: 1,986,141 train and 220,680 validation entities, 5.6% singletons in each. |
| Old blocker (key-based) | 78.9% recall, ~10,550 candidates per entity (median), best possible F₀.₅ 0.910. Would write ~320 GB on test. |
| **Blocker (TF-IDF nearest neighbours)** | **96.4% recall, 44.6 candidates per entity, best possible F₀.₅ 0.988** on validation. Full train takes 17 min at 4.3 GB peak. |
| **Matcher (LightGBM, 42 features)** | Trained on 8.39M pairs from 10% of train entities; 1,079 rounds. |
| **Decision (one owner + threshold 0.75)** | **Validation macro F₀.₅ 0.9696**, link precision 99.3%, recall 92.8%, singletons 0.971. |

Key data facts behind the design:
- Every Source 2/3 record belongs to at most one Source 1 entity (7,638,365 links, all distinct).
- 15% / 12% of Source 2 / Source 3 names are in Indian scripts, while Source 1 is all ASCII.
- No link crosses countries.

## Risks

- **One-owner effect is under-measured.** On validation, only validation entities compete for each record (10% of the pool). The rule's effect on test, where everyone competes, is expected to be larger but is not yet measured.
- **Only 10% of training entities were used for fitting.**
- **France is unseen.** Features and blocking are language-agnostic and country is not a model feature, but France has no labels to check against.
- **Misses split evenly.** 27.3k missed links were never candidates (Indian Source 3, Indian-script names, empty addresses), and 28.0k were rejected by the model (house-number drift, missing target address).

## Next steps (by expected gain)

1. Score every training entity, so the one-owner competition is complete on validation. Then re-tune the threshold.
2. Train on 30–100% of the training entities.
3. Add record-side competition features: how many Source 1 entities a record could go to, and the gap to its second-best entity.
4. Raise k for Indian and Indian-script queries to recover blocking misses.

## Memory rule

Every stage streams in chunks. Heavy runs are started under a watchdog that kills the job above 12 GB. An earlier unbounded version of the blocker exhausted RAM.
