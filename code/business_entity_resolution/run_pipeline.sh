#!/bin/bash
# Stage 1 v3 end to end: blocking (forward + reverse + embedding) -> merge -> record store ->
# pair features -> 3-fold LightGBM (OOF) -> decision tuned on OOF -> output.
#
# Usage: ./run_pipeline.sh split   # entity-level 90/10 train/validation split of the labels
#        ./run_pipeline.sh train   # needs the lab configuration ($WORK_DIR/lab/select.config.json)
#        ./run_pipeline.sh test    # writes output/candidate_pairs.tsv and output/matching_results.tsv
#        ./run_pipeline.sh all
#
# The candidate configuration (pass depths, w, embedding text form) comes from run_lab.sh; the
# decision rule comes from the out-of-fold report of the train stage. Nothing is tuned on test.
# Every stage streams its input in chunks; workers default to all cores (WORKERS=...).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/src"
DATA_DIR="${DATA_DIR:-$HERE/../../student_resource/dataset}"
WORK_DIR="${WORK_DIR:-$HERE/../../work}"
OUTPUT_DIR="${OUTPUT_DIR:-$HERE/../../output}"
CONFIG="${CONFIG:-$WORK_DIR/lab/select.config.json}"
MODEL_DIR="${MODEL_DIR:-$WORK_DIR/model3}"
E5_DIR="${E5_DIR:-}"
PY="${PYTHON:-python3}"
export WORKERS="${WORKERS:-$(nproc 2>/dev/null || echo 4)}"

stage() { echo "[$(date +%H:%M:%S)] === $* ==="; }
cfg() { $PY -c "import json,sys; c=json.load(open('$CONFIG')); print(c.get('$1', $2))"; }

blocking() {  # $1 = split
  local split="$1" w passes
  w="$(cfg _w 0.6)"
  passes="$($PY "$SRC/merge_candidates.py" passes --config "$CONFIG")"
  stage "forward blocking ($split): $passes"
  $PY "$SRC/blocking_knn.py" --data-dir "$DATA_DIR" --split "$split" --index-dir "$WORK_DIR/index3/$split" \
    --passes "$passes" --pairs-out "$WORK_DIR/forward/$split" --report "$WORK_DIR/forward/$split.json"
  local extra=()
  if [ -n "$($PY -c "import json; c=json.load(open('$CONFIG')); print('y' if any(k.startswith('rev_') and v for k,v in c.items()) else '')")" ]; then
    [ -d "$WORK_DIR/reverse/$split" ] || "$HERE/run_lab.sh" reverse "$w" "$split"
    extra+=(--reverse "$WORK_DIR/reverse/$split")
  fi
  if [ "$(cfg emb_rank 0)" -gt 0 ]; then
    local form emb
    form="$(cfg emb_form "'name'")"
    emb="$WORK_DIR/embedding/${split}_$form"
    [ "$split" = train ] && emb="$WORK_DIR/lab/embedding_$form"
    [ -d "$emb" ] || "$HERE/run_lab.sh" embed "$split" "$form"
    extra+=(--embedding "$emb")
  fi
  stage "merge candidates ($split)"
  $PY "$SRC/merge_candidates.py" merge --config "$CONFIG" --forward "$WORK_DIR/forward/$split" "${extra[@]}" \
    --source1 "$DATA_DIR/$split/${split}_source1.tsv" --out-dir "$WORK_DIR/candidates3/$split" \
    --tsv "$WORK_DIR/candidates3/${split}_candidate_pairs.tsv" --work-dir "$WORK_DIR"
}

features() {  # $1 = split, $2 = extra args
  local split="$1"
  stage "record store ($split)"
  $PY "$SRC/record_store.py" --data-dir "$DATA_DIR" --split "$split" --database "$WORK_DIR/records3_$split.sqlite"
  stage "pair features ($split)"
  $PY "$SRC/pair_features.py" --candidates "$WORK_DIR/candidates3/$split" --database "$WORK_DIR/records3_$split.sqlite" \
    --index-dir "$WORK_DIR/index3/$split" --out-dir "$WORK_DIR/features3/$split" --w "$(cfg _w 0.6)" ${2:-}
}

run_split() {
  stage "split train labels into train/validation"
  $PY "$SRC/split.py" --ground-truth "$DATA_DIR/train/train_ground_truth.tsv" --out-dir "$WORK_DIR/split"
}

run_train() {
  blocking train
  features train "--ground-truth $DATA_DIR/train/train_ground_truth.tsv"
  stage "3-fold OOF LightGBM"
  $PY "$SRC/train_matcher.py" --features-dir "$WORK_DIR/features3/train" --truth "$DATA_DIR/train/train_ground_truth.tsv" \
    --model-dir "$MODEL_DIR" ${TRAIN_ARGS:-}
}

run_test() {
  mkdir -p "$OUTPUT_DIR"
  blocking test
  cp "$WORK_DIR/candidates3/test_candidate_pairs.tsv" "$OUTPUT_DIR/candidate_pairs.tsv"
  features test
  stage "score test pairs (mean of fold models)"
  $PY "$SRC/predict.py" --features-dir "$WORK_DIR/features3/test" --model-dir "$MODEL_DIR" \
    --scores-out "$MODEL_DIR/test_scores.parquet" --floor 0.05
  read -r threshold top one_owner < <($PY -c "import json; d=json.load(open('$MODEL_DIR/report.json'))['oof']; \
print(d['threshold'], d['top_threshold'], int(d['one_owner']))")
  stage "decide (OOF-tuned: threshold $threshold, top $top, one-owner $one_owner) -> matching_results.tsv"
  local flag=""
  [ "$one_owner" = 0 ] && flag="--no-one-owner"
  $PY "$SRC/decide.py" --scores "$MODEL_DIR/test_scores.parquet" --source1 "$DATA_DIR/test/test_source1.tsv" \
    --threshold "$threshold" --top-threshold "$top" $flag --output "$OUTPUT_DIR/matching_results.tsv"
  stage "official validator"
  $PY "$DATA_DIR/../utils/validate_submission.py" --matching "$OUTPUT_DIR/matching_results.tsv" \
    --candidate "$OUTPUT_DIR/candidate_pairs.tsv" --test-dir "$DATA_DIR/test"
}

case "${1:-}" in
  split) run_split ;;
  train) run_train ;;
  test) run_test ;;
  all) run_split; run_train; run_test ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
stage "done"
