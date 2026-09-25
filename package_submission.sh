#!/bin/bash
# Build <team_name>_submission.zip in the layout the challenge requires.
# Usage: ./package_submission.sh <team_name>
set -euo pipefail
TEAM="${1:?usage: $0 <team_name>}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

for f in output/matching_results.tsv output/candidate_pairs.tsv Documentation_template.md; do
  [ -f "$ROOT/$f" ] || { echo "missing $f" >&2; exit 1; }
done
mkdir -p "$STAGE/output" "$STAGE/code"
cp "$ROOT/output/matching_results.tsv" "$ROOT/output/candidate_pairs.tsv" "$STAGE/output/"
cp "$ROOT/Documentation_template.md" "$STAGE/"
rsync -a --exclude "__pycache__" --exclude ".DS_Store" "$ROOT/code/business_entity_resolution" "$STAGE/code/"

mkdir -p "$ROOT/dist"
rm -f "$ROOT/dist/${TEAM}_submission.zip"
(cd "$STAGE" && zip -qr "$ROOT/dist/${TEAM}_submission.zip" output code Documentation_template.md)
unzip -l "$ROOT/dist/${TEAM}_submission.zip" | tail -1
echo "wrote dist/${TEAM}_submission.zip"
