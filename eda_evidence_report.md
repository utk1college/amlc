# ML Challenge 2026: Business Entity Resolution — EDA & Evidence Report

**Generated:** 2026-09-25T18:20:46.533207  
**Data Directory:** `E:\Projects\amlc\6ab10eb3b23ba_student_resource\student_resource\dataset`  

## 1. Executive Summary & Actionable Evidence for AI Agents

- 1. STRICT COUNTRY PARTITIONING: Train ground truth shows ZERO cross-country matches.
- 2. OPEN SET GENERALIZATION (FRANCE): Test set introduces France which does not exist in train.
- 3. SINGLETON IMPACT ON MACRO F_0.5: Substantial percentage of Source 1 entities have no matches. Correctly outputting empty strings earns 1.0 macro score.
- 4. MULTI-PASS BLOCKING IS MANDATORY: The MEASURED union recovers 78.17% recall. Marginal (uniquely-contributed) recall per key: first3=0.45%, firstword=0.07%, soundex=0.55%, postal=0.0%.
- 5. NOISE RESILIENCE: Normalizing legal suffixes and token sort ratios drastically increases similarity scores.
- 6. ADDRESS LANDMARKS & PARTIALITY: Address matching in India relies heavily on landmark prepositions. Relying solely on exact postal match will fail.
- 7. SUBMISSION VALIDATION: Validate outputs with utils/validate_submission.py to guarantee strict TSV formatting.

---

