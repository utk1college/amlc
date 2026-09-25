#!/usr/bin/env python3
"""Exact challenge metric: macro F-0.5 over Source-1 entities, singletons included.

Every entity in the truth file is scored. A missing prediction row counts as an
empty prediction; prediction rows for entities absent from the truth file are
ignored, so a full-train prediction file can be scored on the validation split.
"""

import argparse
import json
import sys


def read_links(path):
    """Read a two-column ID-list TSV (matching, candidate or ground-truth file)."""
    links = {}
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if len(header) != 2 or header[0] != "source1_entity_id":
            raise ValueError(f"{path}: unexpected header {header}")
        for line_number, line in enumerate(handle, start=2):
            source1_id, separator, ids = line.rstrip("\n").partition("\t")
            if not separator:
                raise ValueError(f"{path}: row {line_number} has no tab")
            if source1_id in links:
                raise ValueError(f"{path}: duplicate row for {source1_id}")
            links[source1_id] = {x for x in ids.split(",") if x}
    return links


def entity_f05(predicted, truth):
    if not truth:
        return 1.0 if not predicted else 0.0
    hits = len(predicted & truth)
    if not hits:
        return 0.0
    precision, recall = hits / len(predicted), hits / len(truth)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def score(predictions, truth):
    total = singleton_total = singletons = 0.0
    hits = predicted_links = true_links = missing_rows = 0
    for source1_id, true_ids in truth.items():
        predicted = predictions.get(source1_id)
        if predicted is None:
            missing_rows += 1
            predicted = set()
        value = entity_f05(predicted, true_ids)
        total += value
        if not true_ids:
            singletons += 1
            singleton_total += value
        hits += len(predicted & true_ids)
        predicted_links += len(predicted)
        true_links += len(true_ids)
    entities = len(truth)
    non_singletons = entities - singletons
    return {
        "macro_f05": round(total / entities, 6) if entities else None,
        "entities": entities,
        "singletons": int(singletons),
        "singleton_f05": round(singleton_total / singletons, 6) if singletons else None,
        "non_singleton_f05": round((total - singleton_total) / non_singletons, 6) if non_singletons else None,
        "link_precision": round(hits / predicted_links, 6) if predicted_links else None,
        "link_recall": round(hits / true_links, 6) if true_links else None,
        "missing_prediction_rows": missing_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Score predictions with the challenge's macro F-0.5.")
    parser.add_argument("--pred", required=True, help="matching_results.tsv-style predictions")
    parser.add_argument("--truth", required=True, help="ground-truth TSV, e.g. the validation split")
    args = parser.parse_args()
    print(json.dumps(score(read_links(args.pred), read_links(args.truth)), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
