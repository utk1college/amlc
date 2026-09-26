#!/bin/bash
# Blocking lab steps for Stage 1 v3 (train split only; the 10% validation is never queried).
#
#   ./run_lab.sh dump                 v3 forward index + top-100 dump for lab entities, then choose w
#   ./run_lab.sh reverse <w> [split]  reverse pass (Source 2/3 -> Source 1) for a whole split
#   ./run_lab.sh select <budget> <w>  greedy selection on block_dev -> $WORK_DIR/lab/select.config.json
#   ./run_lab.sh confirm              gate 1 on block_confirm against the v2 candidates
#   ./run_lab.sh oracle               what the remaining block_dev misses look like
#   ./run_lab.sh export               e2e_confirm candidate TSVs (v3 and v2) for gate 2
#
# Embedding pairs are picked up from $WORK_DIR/lab/embedding_{name,name_address} when present.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/src"
DATA_DIR="${DATA_DIR:-$HERE/../../student_resource/dataset}"
WORK_DIR="${WORK_DIR:-$HERE/../../work}"
V2_CANDIDATES="${V2_CANDIDATES:-$WORK_DIR/candidates/train_candidates.tsv}"
PY="${PYTHON:-python3}"
LAB="$WORK_DIR/lab"
CONFIG="$LAB/select.config.json"
mkdir -p "$LAB"

stage() { echo "[$(date +%H:%M:%S)] === $* ==="; }
optional() {  # extra lab arguments for the passes that exist
  local args=()
  [ -d "$WORK_DIR/reverse/train" ] && args+=(--reverse "$WORK_DIR/reverse/train")
  [ -d "$LAB/embedding_name" ] && args+=(--embedding "$LAB/embedding_name")
  [ -d "$LAB/embedding_name_address" ] && args+=(--embedding-alt "$LAB/embedding_name_address")
  echo "${args[@]:-}"
}

case "${1:-}" in
  dump)
    stage "v3 forward index + lab dump (top 100 per pass)"
    $PY "$SRC/blocking_knn.py" --data-dir "$DATA_DIR" --split train --index-dir "$WORK_DIR/index3/train" \
      --lab-only --pairs-out "$LAB/forward" --report "$LAB/forward.json" \
      --passes "name:100,address:100,address_char:100,combined@0.4:100,combined@0.5:100,combined@0.6:100,combined@0.7:100"
    stage "choose w on block_dev"
    $PY "$SRC/blocking_lab.py" choose-w --data-dir "$DATA_DIR" --forward "$LAB/forward" --out "$LAB/choose_w.json"
    ;;
  reverse)
    split="${3:-train}"
    stage "reverse pass ($split, w=$2)"
    $PY "$SRC/blocking_knn.py" --data-dir "$DATA_DIR" --split "$split" --direction reverse \
      --index-dir "$WORK_DIR/index3/${split}_s1" --pairs-out "$WORK_DIR/reverse/$split" \
      --report "$WORK_DIR/reverse/$split.json" --passes "name:10,address:10,combined@$2:10"
    ;;
  select)
    stage "greedy selection on block_dev (budget $2, w $3)"
    # shellcheck disable=SC2046
    $PY "$SRC/blocking_lab.py" select --data-dir "$DATA_DIR" --forward "$LAB/forward" $(optional) \
      --budget "$2" --w "$3" --out "$LAB/select.json"
    ;;
  confirm)
    stage "gate 1 on block_confirm"
    # shellcheck disable=SC2046
    $PY "$SRC/blocking_lab.py" confirm --data-dir "$DATA_DIR" --forward "$LAB/forward" $(optional) \
      --config "$CONFIG" --v2 "$V2_CANDIDATES" --out "$LAB/confirm.json"
    ;;
  oracle)
    stage "miss oracle on block_dev"
    # shellcheck disable=SC2046
    $PY "$SRC/blocking_lab.py" oracle --data-dir "$DATA_DIR" --forward "$LAB/forward" $(optional) \
      --config "$CONFIG" --index-dir "$WORK_DIR/index3/train" --out "$LAB/oracle.json"
    ;;
  export)
    stage "e2e_confirm candidates (v3 and v2)"
    # shellcheck disable=SC2046
    $PY "$SRC/blocking_lab.py" export --data-dir "$DATA_DIR" --forward "$LAB/forward" $(optional) \
      --config "$CONFIG" --tsv "$LAB/e2e_v3.tsv" --out "$LAB/export_v3.json"
    $PY "$SRC/blocking_lab.py" export --data-dir "$DATA_DIR" --v2 "$V2_CANDIDATES" \
      --tsv "$LAB/e2e_v2.tsv" --out "$LAB/export_v2.json"
    ;;
  *) sed -n '2,12p' "$0"; exit 2 ;;
esac
stage "done"
