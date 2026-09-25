#!/bin/bash
# End-to-end pipeline: data -> blocking -> record store -> pair features -> matcher -> output.
#
# Usage: ./run_pipeline.sh train   # fit the matcher and report validation macro F0.5
#        ./run_pipeline.sh test    # write output/candidate_pairs.tsv and output/matching_results.tsv
#        ./run_pipeline.sh all
#
# Every stage streams its input in chunks; peak memory stays near 6 GB.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/src"
DATA_DIR="${DATA_DIR:-$HERE/../../student_resource/dataset}"
WORK_DIR="${WORK_DIR:-$HERE/../../work}"
OUTPUT_DIR="${OUTPUT_DIR:-$HERE/../../output}"
MODEL="${MODEL:-$WORK_DIR/model/lgbm.txt}"
THRESHOLD="${THRESHOLD:-0.75}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.10}"
PY="${PYTHON:-python3}"

stage() { echo "[$(date +%H:%M:%S)] === $* ==="; }

run_train() {
  stage "split train labels into train/validation"
  $PY "$SRC/split.py" --ground-truth "$DATA_DIR/train/train_ground_truth.tsv" --out-dir "$WORK_DIR/split"
  stage "blocking (train)"
  $PY "$SRC/blocking_knn.py" --data-dir "$DATA_DIR" --split train --index-dir "$WORK_DIR/index/train" \
    --output "$WORK_DIR/candidates/train_candidates.tsv" --report "$WORK_DIR/candidates/train_report.json"
  stage "record store (train)"
  $PY "$SRC/record_store.py" --data-dir "$DATA_DIR" --split train --database "$WORK_DIR/records_train.sqlite"
  stage "pair features (train sample + validation)"
  $PY "$SRC/pair_features.py" --candidates "$WORK_DIR/candidates/train_candidates.tsv" \
    --database "$WORK_DIR/records_train.sqlite" --index-dir "$WORK_DIR/index/train" \
    --out-dir "$WORK_DIR/features/train" --ground-truth "$DATA_DIR/train/train_ground_truth.tsv" \
    --train-fraction "$TRAIN_FRACTION"
  stage "train matcher"
  $PY "$SRC/train_matcher.py" --features-dir "$WORK_DIR/features/train" --model-out "$MODEL" \
    --scores-out "$WORK_DIR/model/val_scores.parquet"
  stage "validation macro F0.5 by decision threshold"
  $PY "$SRC/decide.py" --scores "$WORK_DIR/model/val_scores.parquet" --truth "$WORK_DIR/split/val_ground_truth.tsv"
}

run_test() {
  mkdir -p "$OUTPUT_DIR"
  stage "blocking (test) -> candidate_pairs.tsv"
  $PY "$SRC/blocking_knn.py" --data-dir "$DATA_DIR" --split test --index-dir "$WORK_DIR/index/test" \
    --output "$OUTPUT_DIR/candidate_pairs.tsv" --report "$WORK_DIR/candidates/test_report.json"
  stage "record store (test)"
  $PY "$SRC/record_store.py" --data-dir "$DATA_DIR" --split test --database "$WORK_DIR/records_test.sqlite"
  stage "pair features (test)"
  $PY "$SRC/pair_features.py" --candidates "$OUTPUT_DIR/candidate_pairs.tsv" \
    --database "$WORK_DIR/records_test.sqlite" --index-dir "$WORK_DIR/index/test" \
    --out-dir "$WORK_DIR/features/test"
  stage "score test pairs"
  $PY "$SRC/predict.py" --features-dir "$WORK_DIR/features/test" --model "$MODEL" \
    --scores-out "$WORK_DIR/model/test_scores.parquet"
  stage "decide -> matching_results.tsv"
  $PY "$SRC/decide.py" --scores "$WORK_DIR/model/test_scores.parquet" \
    --source1 "$DATA_DIR/test/test_source1.tsv" --threshold "$THRESHOLD" \
    --output "$OUTPUT_DIR/matching_results.tsv"
  stage "official validator"
  $PY "$DATA_DIR/../utils/validate_submission.py" --matching "$OUTPUT_DIR/matching_results.tsv" \
    --candidate "$OUTPUT_DIR/candidate_pairs.tsv" --test-dir "$DATA_DIR/test"
}

case "${1:-}" in
  train) run_train ;;
  test) run_test ;;
  all) run_train; run_test ;;
  *) echo "usage: $0 train|test|all" >&2; exit 2 ;;
esac
stage "done"
