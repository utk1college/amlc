# Project Status — Business Entity Resolution

**Date:** 2026-09-25  
**Status:** Candidate-generation implementation complete; matching phase not started

## Objective and constraints

The system must map every Source 1 test record to zero or more matching Source 2/3 records. Evaluation is macro F₀.₅ per Source 1 entity; precision is weighted more heavily than recall, and correct empty predictions for singletons receive full credit. Outputs must be strict TSV, support unseen countries such as France, use no external data or services, and use a permitted model of no more than eight billion parameters.

## Completed work

The supplied challenge context, dataset layout, validator, documentation template, EDA script, and evidence reports were reviewed. The official challenge bundle is isolated under `amazon_shared/`; its contents are not modified and the directory is ignored by Git because teammates already have the dataset.

The EDA established:

- 2,206,821 training Source 1 records and 7,638,365 labelled links.
- 123,247 Source 1 singletons (5.5848%).
- Zero cross-country labelled links.
- France appears in test data but not training data.
- Postal extraction succeeds on approximately 0.1% of sampled rows.
- The sampled union of four simple blocking keys recovered 78.17% of labelled pairs.

The EDA also contained hard-coded reduction-ratio estimates. They were not empirical measurements and are no longer treated as evidence; the correction is documented in `EDA_AUDIT.md`.

`code/business_entity_resolution/src/candidate_generation.py` now implements a streaming SQLite-backed index over Source 2 and Source 3. It unions same-country blocks on normalized-name first three characters, first token, Soundex, and conditional postal/PIN code. It supports arbitrary country labels, deduplicates and sorts IDs, emits one row per Source 1 entity, and reports actual candidate counts and reduction. Its standard-library smoke test passes.

macOS archive metadata, `.DS_Store` files, and Python cache artefacts were removed. The repository-level `.gitignore` excludes the supplied bundle, generated outputs, temporary indexes, reports, and local environments.

## Risks and blockers

There is no final matcher or `matching_results.tsv` yet. The blocker has not been run over the full training data, so its operational recall and reduction are unknown.

The current normalization follows the original EDA and removes non-ASCII characters. This may discard useful information in Indian-language business names. Unicode-preserving normalization and transliteration-independent keys must be compared using only training labels.

The 78.17% value is a sampled EDA result, not a guarantee for the generated candidate file.

## Proposed next steps

1. Run the training blocker with `--evaluate-ground-truth`; record labelled-pair recall, candidate count, and reduction.
2. Compare ASCII and Unicode-preserving name keys, plus address token/numeric keys, using held-out training evidence.
3. Build positive and hard-negative candidate examples without validation leakage.
4. Compute name, address, postal, country, missingness, and block-provenance features.
5. Train a permitted precision-oriented pair classifier and select its threshold by exact macro F₀.₅ on held-out entities.
6. Add entity-level one-to-many and conservative singleton decisions.
7. Generate both output TSVs, enforce that final matches are candidates, and run the official validator.
8. Complete the methodology document and assemble the final package without the ignored dataset.

## Definition of done

Completion requires measured blocker recall, held-out macro F₀.₅ evidence for the matcher, validator-passing output files, reproducible instructions, and a Git repository containing no challenge dataset or generated submission artefacts.
