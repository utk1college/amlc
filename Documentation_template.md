# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** [Your Team Name]  
**Team Members:** [List all team members]  
**Submission Date:** 2026-09-26

---

## 1. Executive Summary

We resolve entities in three stages:
1. **Script-independent normalization and TF-IDF nearest-neighbour blocking.** This keeps 96.4% of true links with about 45 candidates per entity. Our first key-based blocker reached 78.9% with about 10,500.
2. **A LightGBM pair classifier** over about 42 string, token-rarity and house-number features.
3. **A decision layer built on a property of the data:** every Source 2/3 record belongs to at most one Source 1 entity. Its thresholds are tuned directly on macro F0.5.

Validation macro F0.5 is **0.970** on 220,680 held-out training entities.

Three techniques make this work:
- A single offset table transliterates all seven Indian scripts in the data, and a consonant skeleton lets transliterated and English spellings meet.
- A sparse nearest-neighbour search whose cost per query is capped by a posting budget, run chunk by chunk from disk.
- The one-owner decision rule.

---

## 2. Methodology

### 2.1 Problem Analysis

All figures were measured on the training data:

- **Link structure.** There are 2,206,821 Source 1 entities and 7,638,365 labelled links. The links use 7,638,365 *distinct* Source 2/3 IDs, so every Source 2/3 record belongs to at most one Source 1 entity. 26% of Source 2/3 records (2.68M of 10.32M) match nothing and act as distractors.
- **Singletons.** 5.6% of entities have no match. Links per entity range from 0 to more than 10, with a median of 3–4. Predicting all-empty lists scores 0.056. An empty list on an entity that has matches scores 0, so being strict per *link* matters more than predicting empty per *entity*.
- **Countries.** No labelled link crosses countries. Test adds **France** (15% of test Source 1), which training does not contain.
- **Scripts.** 15.1% of Source 2 names and 11.6% of Source 3 names are written in Indian scripts: Devanagari, Telugu, Kannada, Tamil, Bengali, Gujarati and Malayalam. Source 1 names are 100% ASCII. For example, `लक्ष्मी कंसल्टेंसी प्राइवेट लिमिटेड` matches *Lakshmi Consultancy Private Limited*. Keeping Unicode as-is cannot match these, because the two sides share no characters, so they must be transliterated.
- **Name noise:**
  - word reordering (*Empire Inc Translational Interstate*)
  - legal-form swaps and wrappers (`-- `, `<<`, `[LLC]`, `(Private)`)
  - domains used as names (`porternall.com`)
  - typos (*Nioa* / *Noia*)
  - trade names completely unrelated to the registered name
- **Address noise:**
  - house-number drift (2736 → 2683)
  - state code vs full name (*IN* / *Indiana*, *MH* / *Maharashtra*)
  - wrong abbreviation expansion (*St* → *SAINT*)
  - reordered components
  - landmark references (*Nr. City Hall*)
  - empty addresses (3.3% of Source 2/3)
- **Postal codes.** They can be extracted from only about 0.1% of addresses, so they cannot be a primary key.

### 2.2 Solution Strategy

**Approach Type:** Blocking + classifier, followed by a structured decision layer.

**Core Innovation:**
1. Brahmic transliteration through one offset table, plus a consonant skeleton, so that *लिमिटेड*, *Ltd.* and *Limited* all reduce to `lmtd`.
2. Nearest-neighbour blocking with three complementary rankings. Its cost per query is bounded and it streams from an on-disk index.
3. The one-owner rule: each Source 2/3 record goes only to its best-scoring Source 1 entity.

The pipeline follows the multi-stage pattern of the winning Foursquare Location Matching (Kaggle) solutions: generate candidates by nearest neighbours, score them with gradient boosting, then post-process. We adapted it to name and address data without coordinates.

---

## 3. Candidate Generation (Blocking)

**Normalization** (`text_normalize.py`):
- NFC/NFKD, accent stripping and lowercasing.
- Brahmic transliteration, including schwa deletion and Malayalam chillu letters.
- Stripping domain suffixes, splitting on punctuation, and expanding legal forms (`pvt` → `private`, `ltd` → `limited`, …).
- Canonical address abbreviations (`street` / `saint` / `str` → `st`, `road` → `rd`, …).
- A **consonant skeleton** that removes vowels and h, merges aspirates and c/k/s, and collapses repeated letters. For example, *kansaltensi* and *consultancy* both become `knsltns`.

**Representation.** Features are hashed into 2²² buckets, weighted by TF-IDF with IDF computed per country from Source 2/3, and L2-normalized.
- *Name:* character 3-grams of the normalized name, character 3-grams of its skeleton, skeleton words, and skeleton word bigrams.
- *Address:* skeleton words and word bigrams.

**Blocking keys used:** Search always stays within the same country label. Country is treated as an open set, and each label gets its own index. Three top-k rankings are unioned:

| Ranking | k | Why |
| --- | --- | --- |
| Name cosine | 20 | Landmark or missing addresses |
| Address cosine | 20 | Trade names and badly transliterated names |
| 0.6·name + 0.4·address | 30 | The common case; strongest alone |

**Bounded cost.** Each query searches only with its rarest features, and stops once their document frequencies add up to 50,000. So a query can touch at most 50k records, however generic its name is. An earlier version without this cap exhausted RAM on generic names such as *Sri Sai Traders*.

**Chunked execution.**
- Source 2/3 are streamed into 250k-row chunks per country, featurized, and written to disk.
- Source 1 is streamed in batches of 20k.
- Workers score each batch against one index chunk at a time and merge a running top-k.
- Peak memory is 4.3 GB. The full training split (2.2M queries against 10.3M records) takes 17 minutes.

**Candidate pairs generated:**
- Train: 98,407,115 (44.6 per entity)
- Test: 76,965,815 (44.4 per entity; France gets 259,452 entities searched against its own index)

**How we ensured true matches were not lost.** We measured every ranking on held-out entities, both recall@k per ranking and the union, and picked k from those curves. We also check recall by country and by script, so the transliteration gains are visible.

| Blocker | Link recall | Best possible macro F0.5 | Candidates / entity |
| --- | --- | --- | --- |
| Key-based (first 3 characters, first token, Soundex, postal code) | 78.9% | 0.910 | 10,550 (median) |
| **TF-IDF nearest neighbours (final)** | **96.4%** | **0.988** | **44.6** |

On a 5,000-entity validation sample, recall is 94.8% for India and 97.4% for the US. For non-ASCII target names it is 92.9%, up from 43.9% with the key-based blocker.

---

## 4. Matching Model

**Features used** (`pair_features.py`, 42 in total):
- **Name features:**
  - RapidFuzz ratio, partial ratio, token-sort and token-set ratios, and Jaro-Winkler on normalized names.
  - Ratio and token-set ratio on skeletons.
  - IDF-weighted skeleton-word Jaccard, and the IDF sum of shared words.
  - The rarest word present on only one side, measured from each side.
  - Exact match of the core name (legal forms removed), and whether the first core words are equal.
  - Word counts.
  - Legal-form presence on each side, and a conflict flag when both have legal forms that don't overlap.
- **Address features:**
  - Ratio, partial ratio and token-set ratio on normalized addresses, and token-set ratio on skeletons.
  - IDF-weighted address Jaccard, shared IDF, and the rarest one-sided address word from each side.
  - Empty-address flags.
  - Number-set Jaccard, whether the first house numbers are equal, and log |difference| of the first house numbers.
- **Other:**
  - Whether the target name was in an Indian script, and whether the target is from Source 3.
  - A blend of name and address token-set scores. Its rank and gap to the best within the entity's candidate list, the same for name and address token-set, the list size, and how many candidates have a strong name match.
  - Country is **not** a feature, so France is handled the same way.

**Model type:** LightGBM binary classifier (MIT license).
- Settings: 255 leaves, learning rate 0.05, feature and bagging fraction 0.8.
- Early stopping selected 1,079 rounds, using 5% of training entities.
- Trained on 8.39M pairs from 10% of the non-validation entities; 7.5% are positives.
- The features with the highest gain are the name/address blend, house-number gap, rarest missing address word, blend rank, Jaro-Winkler, rarest extra name word, address IDF-Jaccard, and blend gap to best.

**Decision rules** (`decide.py`):
1. **One owner.** Each Source 2/3 record keeps only its highest-scoring Source 1 candidate. This holds for 100% of training links.
2. **Threshold.** A pair is kept if its score is at least *t*. We also let an entity's single best pair pass at a lower *t_top*, because an empty list scores 0 on 94% of entities.

**Threshold selection method:** A grid search over *t* ∈ [0.20, 0.90], *t_top* ∈ {*t*, *t*−0.1, *t*−0.2, *t*−0.3}, with one-owner on or off. Every combination is scored with the exact macro F0.5 (singletons included) on the 220,680 validation entities. The best is one-owner, *t* = *t_top* = 0.75.

---

## 5. Results & Error Analysis

**F_0.5 Score (macro):** **0.9696** on validation (220,680 entities, entity-level split, never used for fitting).

**Test submission:** 1,732,544 rows, 3.21 links per entity, 6.1% empty lists. France (unseen in training) looks like the other countries: 3.24 links per entity and 5.6% empty. It passes the official validator, including `--check-ids`. Public leaderboard: [add score].

| | Value |
| --- | --- |
| Link precision / recall | 99.30% / 92.77% |
| Singletons (12,284) | 0.971 |
| Entities with matches | 0.9695 |
| Best possible with these candidates | 0.988 |
| All-empty baseline | 0.056 |

| Decision rule (validation) | Macro F0.5 |
| --- | --- |
| One-owner, *t* = 0.75 | **0.9696** |
| No one-owner, *t* = 0.75 | 0.9693 |
| One-owner, *t* = 0.70 | 0.9695 |
| One-owner, *t* = 0.80 | 0.9692 |

On validation, only validation entities compete for each record, which is 10% of the real pool. So this table understates the one-owner rule's effect on test, where every entity competes.

**Common false positives (wrong merges).** There are 5,011 wrong links, 391 of them on singletons.
1. *Same address, unrelated name*, e.g. *cedar* vs *nationalautomationinnovations* at 117 Mistletoe Dr. The data also contains true matches of exactly this form (trade names, e.g. *tech technologies private limited* vs *rizasyn* at the same address), so these cases are ambiguous by construction.
2. Near-duplicate names at the same address, labelled as different entities (*brightos safety pllc* vs *pllc brightoz safety*). The record most likely belongs to another look-alike Source 1 entity. The one-owner rule resolves that case when every entity competes.
3. Name-only matches to targets with an empty address.

**Common false negatives (missed matches).** 55,253 links are missed:
- **27,294 never became candidates.**
  - Indian records in Source 3, and records in Indian scripts whose transliteration diverges from the English spelling. Tamil is one example, since one Tamil letter covers k/g, t/d and p/b.
  - Targets with an empty address and only part of the name (*howell diamond* vs *howell diamond manufacturing limited*).
- **27,959 were scored below the threshold.**
  - House-number drift (6012 vs 12).
  - A target address that is missing or reduced to the city.
  - Heavily truncated names (*moon cea* vs *moon moon cea incorporated*).

---

## 6. Conclusion

Most of the score comes from two decisions. First, transliterated, bounded nearest-neighbour blocking lifts the recall ceiling from 0.910 to 0.988 while cutting candidates from about 10,500 to about 45 per entity. Second, a structural fact of the data (one owner per record) plus a threshold tuned on the exact metric keeps link precision at 99.3%.

What we learned:
- Measure the blocker before modelling, since its ceiling bounds everything downstream.
- Bound per-query cost explicitly, and stream every stage, on 10M-record data.
- Tune decisions on the real metric, not on log-loss.

Next steps:
- Score every training entity so the one-owner competition on validation is complete.
- Train on more than 10% of entities.
- Add record-side competition features, such as the score gap to the second-best Source 1 entity.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

| Path | Role |
| --- | --- |
| `run_pipeline.sh` | **Entry point.** `./run_pipeline.sh train` fits the model and prints the validation F0.5 grid. `./run_pipeline.sh test` writes `output/candidate_pairs.tsv` and `output/matching_results.tsv`, then runs the official validator. |
| `src/text_normalize.py` | Normalization, transliteration, skeleton |
| `src/blocking_knn.py` | Chunked TF-IDF nearest-neighbour blocking |
| `src/record_store.py` | On-disk normalized record store |
| `src/pair_features.py` | Pair features (Parquet per batch) |
| `src/train_matcher.py`, `src/predict.py` | LightGBM training and scoring |
| `src/decide.py` | One-owner rule, thresholds, output writing |
| `src/split.py`, `src/score.py`, `src/evaluate_candidates.py` | Validation split, exact metric, blocking diagnostics |
| `tests/` | Unit tests for the metric, normalization, blocking contract and decision rules |
| `README.md`, `requirements.txt` | Setup and pinned dependencies |

No external data, APIs, geocoding or pretrained language models are used.

### B. Additional Results

Recall@k for each blocking ranking (5,000 validation entities, share of true links found):

| Ranking | @1 | @5 | @10 | @20 | @50 | @100 |
| --- | --- | --- | --- | --- | --- | --- |
| Name | 15.5% | 44.1% | 52.0% | 57.9% | 65.0% | 70.3% |
| Address | 23.8% | 73.3% | 81.8% | 85.7% | 88.4% | 89.9% |
| Name + address | 25.8% | 85.3% | 93.0% | 94.8% | 96.1% | 96.8% |

The @1 values cannot exceed about 28%: each entity has one top candidate, and the sample has 4,720 entities with matches against 17,123 links.

Test-set run on a 12-core Mac with 24 GB RAM (`./run_pipeline.sh test`):

| Stage | Time | Output |
| --- | --- | --- |
| Index + blocking | 15 min | 76,965,815 candidate pairs |
| Record store | 1.3 min | 11.7M normalized records |
| Pair features | 25 min | 87 Parquet batches |
| LightGBM scoring | 20 min | 5.99M pairs at score ≥ 0.3 |
| Decide + validator | 0.5 min | `matching_results.tsv` |

Peak memory for the whole run was 5.4 GB.

Candidate budget vs recall (union of the three rankings, 5,000 validation entities):

| k (name/address/combined) | Candidates / entity | Link recall | Best possible F0.5 |
| --- | --- | --- | --- |
| 10 / 10 / 20 | 25.5 | 95.7% | 0.985 |
| **20 / 20 / 30 (final)** | **44.7** | **96.4%** | **0.988** |
| 100 / 100 / 100 | 200.4 | 97.7% | 0.992 |
