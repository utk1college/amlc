#!/usr/bin/env python3
"""Read-only diagnostics: where does macro F-0.5 go, and does test look like validation?

`loss` splits the validation error by country, source and target-name script,
and separates links that were never candidates from links the model rejected.

`shift` compares validation with test per country using scores only (no labels):
top-1 score histogram, links per entity, empty-list rate and the share of
targets claimed by two or more entities. Validation holds 10% of the Source-1
entities, so test is also measured on the same 10% hash subsample; that keeps
the amount of competition between entities comparable.

Neither command writes anything except the optional --out JSON.
"""

import argparse
import json
import os
import sys

import polars as pl

from decide import decide, to_links
from evaluate_candidates import read_columns
from score import entity_f05, read_links
from split import is_validation

HISTOGRAM_EDGES = (0.3, 0.5, 0.75, 0.9)


def country_of(path, keep=None):
    """Source-1 ID -> casefolded country label, as the blocker keys it."""
    countries = {}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if keep is None or parts[0] in keep:
                countries[parts[0]] = parts[3].strip().casefold()
    return countries


def pairs_frame(mapping):
    rows = [(s, t) for s, targets in mapping.items() for t in targets]
    return pl.DataFrame({"source1_id": [r[0] for r in rows], "target_id": [r[1] for r in rows]},
                        schema={"source1_id": pl.Utf8, "target_id": pl.Utf8})


def add_slices(frame, countries, target_names):
    """Attach country, source (S2/S3) and target-name script columns."""
    lookup = pl.DataFrame({"source1_id": list(countries), "country": list(countries.values())})
    names = pl.DataFrame({"target_id": list(target_names), "non_ascii": [not n.isascii() for n in target_names.values()]})
    return (frame.join(lookup, on="source1_id", how="left")
            .join(names, on="target_id", how="left")
            .with_columns(pl.col("target_id").str.slice(0, 2).alias("source"),
                          pl.col("non_ascii").fill_null(False)))


def loss_report(scores, truth, countries, target_names, threshold, top_threshold):
    selected = decide(scores, threshold, top_threshold, one_owner=True).select("source1_id", "target_id")
    predicted = to_links(selected)

    candidate = scores.select("source1_id", "target_id").with_columns(pl.lit(True).alias("candidate"))
    chosen = selected.with_columns(pl.lit(True).alias("selected"))
    links = (pairs_frame(truth).join(candidate, on=["source1_id", "target_id"], how="left")
             .join(chosen, on=["source1_id", "target_id"], how="left")
             .with_columns(pl.col("candidate").fill_null(False), pl.col("selected").fill_null(False)))
    links = add_slices(links, countries, target_names).with_columns(
        pl.when(pl.col("selected")).then(pl.lit("tp"))
        .when(pl.col("candidate")).then(pl.lit("rejected"))
        .otherwise(pl.lit("not_candidate")).alias("status"))
    false_links = add_slices(
        selected.join(pairs_frame(truth).with_columns(pl.lit(True).alias("true_link")),
                      on=["source1_id", "target_id"], how="left").filter(pl.col("true_link").is_null()),
        countries, target_names)

    keys = ["country", "source", "non_ascii"]
    by_slice = {}
    for row in links.group_by(keys).agg(
            (pl.col("status") == "tp").sum().alias("tp"),
            (pl.col("status") == "rejected").sum().alias("rejected"),
            (pl.col("status") == "not_candidate").sum().alias("not_candidate")).iter_rows(named=True):
        by_slice["/".join(str(row[k]) for k in keys)] = {k: row[k] for k in ("tp", "rejected", "not_candidate")}
    for row in false_links.group_by(keys).agg(pl.len().alias("false_positive")).iter_rows(named=True):
        by_slice.setdefault("/".join(str(row[k]) for k in keys), {})["false_positive"] = row["false_positive"]

    entities = len(truth)
    per_country = {}
    for source1_id, true_ids in truth.items():
        value = entity_f05(predicted.get(source1_id, set()), true_ids)
        bucket = per_country.setdefault(countries.get(source1_id, "?"),
                                        {"entities": 0, "singletons": 0, "loss": 0.0, "singleton_loss": 0.0})
        bucket["entities"] += 1
        bucket["loss"] += 1 - value
        if not true_ids:
            bucket["singletons"] += 1
            bucket["singleton_loss"] += 1 - value
    for bucket in per_country.values():
        bucket["mean_f05"] = round(1 - bucket["loss"] / bucket["entities"], 5)
        bucket["share_of_total_loss_pts"] = round(100 * bucket["loss"] / entities, 4)
        bucket["singleton_share_of_loss_pts"] = round(100 * bucket["singleton_loss"] / entities, 4)
        bucket["loss"] = round(bucket["loss"], 1)
        bucket["singleton_loss"] = round(bucket["singleton_loss"], 1)
    return {"decision": {"threshold": threshold, "top_threshold": top_threshold, "one_owner": True},
            "by_country": per_country, "links_by_country_source_nonascii": by_slice}


def shift_stats(scores, entities, threshold, top_threshold):
    """Label-free score statistics for the given set of Source-1 entities."""
    scores = scores.filter(pl.col("source1_id").is_in(list(entities)))
    top1 = scores.group_by("source1_id").agg(pl.col("score").max().alias("top1"))
    edges = (0.0,) + HISTOGRAM_EDGES + (1.0001,)
    values = [0.0] * (len(entities) - top1.height) + top1["top1"].to_list()
    histogram = {f"[{lo},{hi if hi < 1 else 1.0})": 0 for lo, hi in zip(edges, edges[1:])}
    for value in values:
        for lo, hi in zip(edges, edges[1:]):
            if lo <= value < hi:
                histogram[f"[{lo},{hi if hi < 1 else 1.0})"] += 1
                break
    total = max(len(entities), 1)
    chosen = decide(scores, threshold, top_threshold, one_owner=True)
    strong = scores.filter(pl.col("score") >= 0.5)
    owners = strong.group_by("target_id").agg(pl.col("source1_id").n_unique().alias("owners"))
    return {
        "entities": len(entities),
        "top1_histogram_share": {k: round(v / total, 4) for k, v in histogram.items()},
        "links_per_entity": round(chosen.height / total, 3),
        "empty_list_share": round(1 - chosen["source1_id"].n_unique() / total, 4),
        "targets_scored_ge_0.5": owners.height,
        "contested_target_share": round((owners["owners"] > 1).mean(), 4) if owners.height else None,
    }


def shift_report(val_scores, test_scores, val_countries, test_countries, floor, threshold, top_threshold):
    val_scores = val_scores.filter(pl.col("score") >= floor)
    test_scores = test_scores.filter(pl.col("score") >= floor)
    subsample = {i for i in test_countries if is_validation(i)}
    report = {"score_floor": floor, "decision": {"threshold": threshold, "top_threshold": top_threshold},
              "note": "test_10pct uses the validation hash on test IDs, so competition matches validation"}
    for country in sorted(set(val_countries.values()) | set(test_countries.values())):
        row = {}
        for name, scores, countries, restrict in (("val", val_scores, val_countries, None),
                                                  ("test_10pct", test_scores, test_countries, subsample),
                                                  ("test_all", test_scores, test_countries, None)):
            entities = {i for i, c in countries.items() if c == country and (restrict is None or i in restrict)}
            if entities:
                row[name] = shift_stats(scores, entities, threshold, top_threshold)
        report[country] = row
    return report


def emit(report, out):
    text = json.dumps(report, indent=2)
    print(text)
    if out:
        with open(out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


def main():
    parser = argparse.ArgumentParser(description="Read-only error and shift diagnostics.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("loss", "shift"):
        p = sub.add_parser(name)
        p.add_argument("--val-scores", required=True, help="val_scores.parquet (source1_id, target_id, score)")
        p.add_argument("--data-dir", required=True, help="Dataset directory with train/ and test/")
        p.add_argument("--threshold", type=float, default=0.75)
        p.add_argument("--top-threshold", type=float)
        p.add_argument("--out")
    sub.choices["loss"].add_argument("--truth", required=True, help="val_ground_truth.tsv")
    sub.choices["shift"].add_argument("--test-scores", required=True)
    sub.choices["shift"].add_argument("--truth", required=True, help="val_ground_truth.tsv (defines validation entities)")
    sub.choices["shift"].add_argument("--floor", type=float, default=0.3,
                                      help="Compare pairs at or above this score; test scores are floored.")
    args = parser.parse_args()
    top = args.top_threshold if args.top_threshold is not None else args.threshold

    train = os.path.join(args.data_dir, "train")
    truth = read_links(args.truth)
    val_scores = pl.read_parquet(args.val_scores, columns=["source1_id", "target_id", "score"])
    val_countries = country_of(os.path.join(train, "train_source1.tsv"), set(truth))
    if args.command == "loss":
        linked = set().union(*truth.values()) | set(val_scores.filter(pl.col("score") >= args.threshold)["target_id"])
        target_names = {}
        for source in ("train_source2.tsv", "train_source3.tsv"):
            target_names.update(read_columns(os.path.join(train, source), linked, 1))
        emit(loss_report(val_scores, truth, val_countries, target_names, args.threshold, top), args.out)
    else:
        test_scores = pl.read_parquet(args.test_scores, columns=["source1_id", "target_id", "score"])
        test_countries = country_of(os.path.join(args.data_dir, "test", "test_source1.tsv"))
        emit(shift_report(val_scores, test_scores, val_countries, test_countries,
                          args.floor, args.threshold, top), args.out)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
