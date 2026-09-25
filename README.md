# Amazon ML Challenge 2026 — Business Entity Resolution

Links each Source 1 business record to its matching Source 2 and Source 3 records, using only the supplied challenge data. The score is macro F₀.₅ per Source 1 entity, with singletons included.

**Current validation macro F₀.₅: 0.970** (220,680 held-out training entities). Full numbers are in `status.md`, and the methodology is in `Documentation_template.md`.

## Repository boundaries

The challenge bundle stays local and is ignored by Git: `student_resource/` here, or `amazon_shared/` on teammates' machines. Never commit data, generated candidates, features, models or submission zips. `.gitignore` excludes `student_resource/`, `amazon_shared/`, `work/`, `output/`, `dist/` and `*.zip`.

## Run

```bash
pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution
./run_pipeline.sh train    # blocking → features → LightGBM → validation F0.5 grid
./run_pipeline.sh test     # writes output/candidate_pairs.tsv + output/matching_results.tsv, runs the validator
```

If your data lives elsewhere, set `DATA_DIR`, for example `DATA_DIR=../../amazon_shared/6ab10eb3b23ba_student_resource/student_resource/dataset`. Every stage streams in chunks, and peak memory is about 6 GB.

Build the final zip from the repository root:

```bash
./package_submission.sh <team_name>
```

## Layout

| Item | Purpose |
| --- | --- |
| `code/business_entity_resolution/` | Runnable pipeline (`run_pipeline.sh`, `src/`, `tests/`, README, pinned requirements) |
| `Documentation_template.md` | Filled-in methodology write-up for the submission |
| `package_submission.sh` | Builds `dist/<team>_submission.zip` in the required layout |
| `status.md` | Measured results, risks and next steps |
| `EDA_AUDIT.md`, `eda_entity_resolution.py`, `eda_evidence_report.*` | Original EDA and its audit |
| `peek_addresses.py` | Small local utility for inspecting addresses; not part of the pipeline |
| `instructions.md` | Copy of the challenge problem statement |

## Tests

```bash
python -m unittest discover -s code/business_entity_resolution/tests -v
```
