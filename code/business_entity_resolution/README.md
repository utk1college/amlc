# Business Entity Resolution pipeline

Links every Source 1 business to its matching Source 2 / Source 3 records using
only the supplied TSV files. Validation macro F0.5 is **0.970** on a held-out 10%
of training entities.

```
data ─► blocking_knn.py ─► record_store.py ─► pair_features.py ─► LightGBM ─► decide.py ─► output/
        TF-IDF nearest       normalized          ~42 similarity     pair        one owner per
        neighbours, ~45      records on disk     features / pair    classifier  record + threshold
        candidates / entity
```

Every stage streams its input in chunks and never loads a whole source into
memory. Peak memory is about 6 GB.

## Setup

Python 3.10+ (developed on 3.13).

```bash
pip install -r requirements.txt
```

The challenge data must sit at `../../student_resource/dataset` (with `train/`
and `test/`), or you can point `DATA_DIR` at it.

## Reproduce end to end

From this directory:

```bash
./run_pipeline.sh train   # ~35 min: blocking, features, LightGBM, validation F0.5 report
./run_pipeline.sh test    # ~60 min: writes ../../output/candidate_pairs.tsv and matching_results.tsv
```

`test` needs the model written by `train` (`../../work/model/lgbm.txt`). The
last test stage runs the official `validate_submission.py`. You can override
paths and settings with environment variables: `DATA_DIR`, `WORK_DIR`,
`OUTPUT_DIR`, `MODEL`, `THRESHOLD` (0.75) and `TRAIN_FRACTION` (0.10).

Intermediate artefacts go to `../../work/`: indexes, record stores, features,
model and scores. They take about 20 GB for both splits.

## Source layout

| File | Role |
| --- | --- |
| `src/text_normalize.py` | Normalizes names and addresses. Transliterates all Brahmic scripts through one offset table, strips legal-form, domain and punctuation noise, expands abbreviations, and builds the consonant skeleton that makes transliterated and English spellings meet. |
| `src/blocking_knn.py` | Candidate generation. Builds a per-country, chunked, on-disk TF-IDF index, runs top-k search by name, by address and by both, and caps search cost per query. |
| `src/record_store.py` | SQLite store of normalized records, fetched by ID per batch. |
| `src/pair_features.py` | Pair features: RapidFuzz string similarity, IDF-weighted token overlap, house-number agreement, and ranks within the entity. |
| `src/train_matcher.py` | Trains LightGBM (MIT license) on non-validation entities and scores validation pairs. |
| `src/predict.py` | Scores test feature batches. |
| `src/decide.py` | Decision rules and exact-metric threshold search. |
| `src/split.py` | Deterministic entity-level 90/10 train/validation split. |
| `src/score.py` | The challenge metric: macro F0.5 with singletons. |
| `src/evaluate_candidates.py` | Candidate recall, candidate-list sizes, and the best F0.5 reachable with those candidates. |
| `src/candidate_generation.py` | Earlier baseline blocker, kept for comparison (78.9% recall, ~10k candidates per entity). |

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Constraints preserved

- No external data, lookups, geocoding or APIs. IDF weights and every learned value come from the supplied files.
- Country is treated as an open set of labels. Each label gets its own index, and no country appears as a model feature, so France runs through the same path.
- `candidate_pairs.tsv` is exactly the set the model scores, so every match is a candidate.
- The model is LightGBM (MIT license), far below the 8B-parameter limit.
