# Amazon ML Challenge 2026 — Business Entity Resolution

This repository contains the reproducible engineering work for linking Source 1 business records to matching records in Source 2 and Source 3 using only supplied challenge data. The scored objective is macro F₀.₅, which weights precision more heavily than recall and includes singleton entities.

## Repository boundaries

The official challenge bundle is kept locally under `amazon_shared/` and is ignored by Git. Its files are not modified. It contains the dataset, validator, challenge instructions, and documentation template. Teammates already have the dataset, so data files must not be committed or pushed.

Tracked files contain the EDA, its audit, and candidate-generation code. Generated candidates, reports, temporary indexes, and caches are ignored.

## Current implementation

The EDA is complete enough to establish the initial implementation gate. It verified the schema, record counts, ground-truth topology, country partitioning, field completeness, and sampled string noise. Hard-coded reduction estimates were removed from active evidence and documented in `EDA_AUDIT.md`.

The implemented next phase is the deterministic SQLite-backed blocker at `code/business_entity_resolution/src/candidate_generation.py`. It unions same-country first-three-name-character, first-token, Soundex, and conditional postal-code blocks. Country labels are open-set, so France is supported without special-case filtering. The generator writes the exact candidates a later matcher must score and emits measured reports.

## Layout

```text
amazon_shared/                         # supplied files; ignored; do not edit
code/business_entity_resolution/       # runnable pipeline and smoke test
EDA_AUDIT.md                            # evidence and limitations
eda_entity_resolution.py                # EDA implementation
eda_evidence_report.json                # EDA evidence
status.md                               # formal status and plan
```

## What each repository item is

| Item | Purpose |
| --- | --- |
| `code/business_entity_resolution/` | Runnable pipeline code and its smoke test. |
| `.gitignore` | Prevents the local challenge data and generated artefacts from being committed. |
| `EDA_AUDIT.md` | Records which EDA findings are reliable and documents the corrected reduction-ratio issue. |
| `README.md` | Project entry point: scope, layout, commands, validation, and next phase. |
| `eda_entity_resolution.py` | Original exploratory data-analysis script used to inspect schemas, linkage topology, noise, and blocking evidence. |
| `eda_evidence_report.json` | Machine-readable quantitative EDA results. |
| `eda_evidence_report.md` | Short human-readable summary of the EDA evidence. |
| `peek_addresses.py` | Small local utility for manually inspecting address examples; it is not part of the production pipeline. |
| `status.md` | Formal project record covering completed work, risks, blockers, and proposed next steps. |
| `amazon_shared/` | Local, Git-ignored copy of the official challenge materials and dataset; do not edit or commit it. |

## Requirements and tests

Candidate generation uses Python 3.8+ standard-library modules only. Run from the repository root:

```powershell
python -m unittest discover -s code/business_entity_resolution/tests -v
```

## Measure the training blocker

The supplied data must exist under `amazon_shared/6ab10eb3b23ba_student_resource/student_resource/dataset`:

```powershell
python code/business_entity_resolution/src/candidate_generation.py `
  --data-dir amazon_shared/6ab10eb3b23ba_student_resource/student_resource/dataset `
  --split train `
  --output output/train_candidate_pairs.tsv `
  --report output/train_blocking_report.json `
  --work-dir work `
  --evaluate-ground-truth
```

This is the required next measurement. For test candidates, use `--split test` and omit `--evaluate-ground-truth`.

## Validate the final submission

After the matching stage is implemented, run the official validator:

```powershell
python amazon_shared/6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir amazon_shared/6ab10eb3b23ba_student_resource/student_resource/dataset/test
```

The final matcher must emit only IDs present in the candidate file, include every test Source 1 ID exactly once, support one-to-many links, and choose conservative empty predictions when evidence is insufficient.

See `status.md` for the formal record of completed work, risks, blockers, and the proposed route to a complete submission.
