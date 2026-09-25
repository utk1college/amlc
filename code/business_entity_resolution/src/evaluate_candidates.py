#!/usr/bin/env python3
"""Measure a candidate file against labelled links.

Reports link recall, candidate-list sizes, and the blocking ceiling: the macro
F-0.5 a perfect matcher would reach if it could only pick from these
candidates. With --data-dir, recall is also broken down by Source-1 country
and by whether the linked record's name is non-ASCII.
"""

import argparse
import json
import os
import sys

from score import entity_f05, read_links


def percentile(sorted_values, fraction):
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1, int(fraction * len(sorted_values)))]


def read_columns(path, wanted_ids, column):
    values = {}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in wanted_ids:
                values[parts[0]] = parts[column]
    return values


def rate(found, total):
    return {"links": total, "recall_pct": round(100 * found / total, 3) if total else None}


def evaluate(candidates, truth, countries=None, target_names=None):
    sizes, ceiling, found, total = [], 0.0, 0, 0
    by_country, by_script = {}, {}
    for source1_id, true_ids in truth.items():
        candidate_ids = candidates.get(source1_id, set())
        sizes.append(len(candidate_ids))
        recovered = true_ids & candidate_ids
        ceiling += entity_f05(recovered, true_ids)
        found += len(recovered)
        total += len(true_ids)
        if countries is not None:
            bucket = by_country.setdefault(countries.get(source1_id, "?"), [0, 0])
            bucket[0] += len(recovered)
            bucket[1] += len(true_ids)
        if target_names is not None:
            for target in true_ids:
                script = "non_ascii" if not target_names.get(target, "").isascii() else "ascii"
                bucket = by_script.setdefault(script, [0, 0])
                bucket[0] += target in candidate_ids
                bucket[1] += 1
    sizes.sort()
    report = {
        "source1_entities": len(truth),
        "labelled_links": total,
        "link_recall_pct": round(100 * found / total, 3) if total else None,
        "ceiling_macro_f05": round(ceiling / len(truth), 6) if truth else None,
        "candidate_pairs": sum(sizes),
        "candidates_per_source1": {
            "mean": round(sum(sizes) / len(sizes), 1) if sizes else None,
            "median": percentile(sizes, 0.5),
            "p90": percentile(sizes, 0.9),
            "p99": percentile(sizes, 0.99),
            "max": sizes[-1] if sizes else None,
            "empty": sum(size == 0 for size in sizes),
        },
    }
    if countries is not None:
        report["recall_by_country"] = {k: rate(*v) for k, v in sorted(by_country.items())}
    if target_names is not None:
        report["recall_by_target_name_script"] = {k: rate(*v) for k, v in sorted(by_script.items())}
    return report


def main():
    parser = argparse.ArgumentParser(description="Measure blocking recall, size and F-0.5 ceiling.")
    parser.add_argument("--candidates", required=True, help="candidate_pairs.tsv-style file")
    parser.add_argument("--truth", required=True, help="ground-truth TSV; only its entities are measured")
    parser.add_argument("--data-dir", help="Dataset directory with train/ for country and script breakdowns")
    args = parser.parse_args()

    truth = read_links(args.truth)
    candidates = read_links(args.candidates)
    countries = target_names = None
    if args.data_dir:
        train = os.path.join(args.data_dir, "train")
        countries = read_columns(os.path.join(train, "train_source1.tsv"), set(truth), 3)
        linked = set().union(*truth.values())
        target_names = {}
        for source in ("train_source2.tsv", "train_source3.tsv"):
            target_names.update(read_columns(os.path.join(train, source), linked, 1))
    print(json.dumps(evaluate(candidates, truth, countries, target_names), indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
