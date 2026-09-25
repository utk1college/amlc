#!/usr/bin/env python3
"""Turn pair scores into match lists, tuned by exact macro F-0.5.

Decision rules, applied in order:
1. One owner: every Source-2/3 record may belong to at most one Source-1
   entity (true for every training link), so each target keeps only its
   highest-scoring Source-1 candidate.
2. A pair is matched when its score reaches --threshold; an entity's single
   best pair needs only --top-threshold, because an empty list scores 0 for
   the 94% of entities that do have matches.

With --truth the script grid-searches both thresholds and reports the best;
with --output it writes matching_results.tsv for every Source-1 ID.
"""

import argparse
import json

import polars as pl

from score import read_links, score


def decide(scores, threshold, top_threshold, one_owner=True):
    frame = scores
    if one_owner:
        frame = frame.filter(pl.col("score") == pl.col("score").max().over("target_id"))
    best = pl.col("score") == pl.col("score").max().over("source1_id")
    return frame.filter((pl.col("score") >= threshold) | (best & (pl.col("score") >= top_threshold)))


def to_links(frame):
    return {s: set(t) for s, t in frame.group_by("source1_id").agg(pl.col("target_id")).iter_rows()}


def grid(scores, truth):
    results = []
    for one_owner in (False, True):
        for step in range(4, 19):
            threshold = step / 20
            for drop in (0.0, 0.1, 0.2, 0.3):
                top = round(max(threshold - drop, 0.05), 2)
                result = score(to_links(decide(scores, threshold, top, one_owner)), truth)
                results.append({"one_owner": one_owner, "threshold": threshold, "top_threshold": top, **result})
    return sorted(results, key=lambda r: -r["macro_f05"])


def main():
    parser = argparse.ArgumentParser(description="Decide matches from pair scores.")
    parser.add_argument("--scores", required=True, help="Parquet with source1_id, target_id, score")
    parser.add_argument("--truth", help="Ground truth for a threshold grid search (validation).")
    parser.add_argument("--source1", help="Source-1 TSV listing every entity to write a row for.")
    parser.add_argument("--output", help="matching_results.tsv to write.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-threshold", type=float)
    parser.add_argument("--no-one-owner", action="store_true")
    args = parser.parse_args()

    scores = pl.read_parquet(args.scores, columns=["source1_id", "target_id", "score"])
    if args.truth:
        results = grid(scores, read_links(args.truth))
        for row in results[:10]:
            print(json.dumps(row))
        best_off = max((r for r in results if not r["one_owner"]), key=lambda r: r["macro_f05"])
        print("best without one-owner:", json.dumps(best_off))
    if args.output:
        top = args.top_threshold if args.top_threshold is not None else args.threshold
        links = to_links(decide(scores, args.threshold, top, not args.no_one_owner))
        with open(args.source1, encoding="utf-8") as source, open(args.output, "w", encoding="utf-8") as out:
            source.readline()
            out.write("source1_entity_id\tmatched_entity_ids\n")
            for line in source:
                source1_id = line.split("\t", 1)[0]
                out.write(f"{source1_id}\t{','.join(sorted(links.get(source1_id, ())))}\n")


if __name__ == "__main__":
    main()
