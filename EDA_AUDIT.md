# EDA audit and implementation gate

## Decision

The foundation is sufficient to implement a **baseline candidate-generation
stage**, but it is not sufficient to claim a final blocking recall ceiling or a
candidate reduction ratio. The baseline is deliberately limited to the four
keys actually measured in the EDA and must be re-evaluated from its generated
training candidates before matching-model work begins.

## Evidence that may be used

- All seven source files have the expected four-column TSV schema and unique
  entity IDs.
- The labelled training links contain 7,638,365 pairs; the EDA found zero
  cross-country links. Country may therefore partition a block, while its
  values must remain arbitrary strings so France is supported at inference.
- Ground truth has 2,206,821 Source-1 rows, of which 123,247 (5.5848%) are
  singletons. Macro F_0.5 consequently requires an explicit conservative
  no-match decision in the later matching stage.
- On the reproducible 50,000 labelled-pair sample, the union of exact-country
  plus normalized-name first-three-character, first-token, Soundex, and
  extracted-postal keys covered 78.17% of sampled true pairs.
- Postal extraction was absent in about 99.9% of the sampled rows. It remains
  a harmless conditional key, not a main blocking dependency.

## Correction to the prior report

The `estimated_reduction_ratio_pct` values in the original EDA source were
constants, not measurements. They must not be cited or used for decisions.
`eda_entity_resolution.py` now records them as `null`; the supplied blocking
script emits measured candidate counts and comparison reduction instead.

The EDA also does not calculate a token-sort ratio despite mentioning it in a
summary sentence. Do not treat that sentence as a measured result.

## Required next evidence

Run the blocker on train with `--evaluate-ground-truth`. Its JSON report must
record labelled-pair recall, candidate-pair count, and reduction ratio. Only
then decide whether to add keys. Any added key must be evaluated on the same
held-out or labelled-train protocol; no external business data, geocoding, or
country-specific hard-coding is allowed.
