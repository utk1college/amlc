# Business Entity Resolution pipeline

This repository currently implements the audited baseline **candidate
generation / blocking** phase. It uses only the supplied TSV files and Python's
standard library.

## Baseline blocking keys

For each Source-1 record, candidates from Source 2 and Source 3 are the union
of these exact keys, always within the same normalized country string:

1. first three characters of the EDA-normalized business name;
2. first token of that normalized name;
3. Soundex of that normalized name;
4. an extracted postal/PIN code, only when one is present.

The country value is normalized generically with `strip().casefold()`; no
country labels are enumerated, so France and future labels remain supported.
The script does not cap or silently remove candidates. The resulting TSV is
therefore exactly the list a subsequent matcher must score.

## Run

From this directory, create baseline test candidates:

```powershell
python src/candidate_generation.py `
  --data-dir ..\\..\\amazon_shared\\6ab10eb3b23ba_student_resource\\student_resource\\dataset `
  --split test `
  --output ..\\..\\output\\candidate_pairs.tsv `
  --report ..\\..\\output\\test_blocking_report.json `
  --work-dir ..\\..\\work
```

Before modelling, measure the blocker on all training labels:

```powershell
python src/candidate_generation.py `
  --data-dir ..\\..\\amazon_shared\\6ab10eb3b23ba_student_resource\\student_resource\\dataset `
  --split train `
  --output ..\\..\\output\\train_candidate_pairs.tsv `
  --report ..\\..\\output\\train_blocking_report.json `
  --work-dir ..\\..\\work `
  --evaluate-ground-truth
```

The train output is an evaluation artefact, not a submission file. The test
output has one row for every test Source-1 ID and only test Source-2/3 IDs.

## Constraints preserved

- Inputs and outputs are read/written as UTF-8 TSV.
- No external data, lookups, geocoding, or APIs are used.
- Candidate IDs are deduplicated and sorted deterministically.
- The final matching stage must only emit IDs from this candidate TSV.
- Validate final `matching_results.tsv` and `candidate_pairs.tsv` with the
  provided `utils/validate_submission.py` before submitting.

## Next phase

Use the measured train blocker report to establish recall and reduction. Build
labelled candidate pairs without leaking an entity across a validation split,
train a precision-oriented pair scorer, and choose per-entity predictions by
macro F_0.5 on validation, including the empty prediction for singletons.
