#!/usr/bin/env python3
"""Robustness checks that need the pair-feature Parquet files (read-only).

`loco` (leave-one-country-out): fit the pair model on non-validation pairs of
one country and score the validation pairs of another, then apply the normal
decision rule. Comparing --train-country X --eval-country Y with the in-domain
X/X and Y/Y runs shows how much of the model is country-specific. France has no
labels, so this is the closest label-based proxy for an unseen country.

`adversarial`: train a classifier to tell validation top-1 candidate pairs from
test top-1 candidate pairs, per country. An AUC near 0.5 means the model sees
similar inputs at test time; a high AUC lists the features that drifted.
Uses only unlabeled feature values.
"""

import argparse
import glob
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from decide import decide, to_links
from pair_features import FEATURE_COLUMNS, V2_FEATURES
from score import read_links, score
from split import is_validation
from train_matcher import PARAMS


def country_of(path):
    countries = {}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            countries[parts[0]] = parts[3].strip().casefold()
    return countries


def batches(features_dir, columns):
    for path in sorted(glob.glob(os.path.join(features_dir, "*.parquet"))):
        yield pl.read_parquet(path, columns=columns)


def loco(args):
    countries = country_of(args.source1)
    truth = {s: t for s, t in read_links(args.truth).items() if countries.get(s) == args.eval_country}
    fit_x, fit_y, eval_frames = [], [], []
    for frame in batches(args.features_dir, ["source1_id", "target_id", "is_validation", "label", *FEATURE_COLUMNS]):
        ids = frame["source1_id"].to_list()
        in_country = np.array([countries.get(i) for i in ids]) == args.train_country
        picked = np.array([is_validation("loco" + i, args.train_fraction) for i in ids])
        validation = frame["is_validation"].to_numpy()
        rows = in_country & picked & ~validation
        if rows.any():
            part = frame.filter(pl.Series(rows))
            fit_x.append(part.select(FEATURE_COLUMNS).to_numpy().astype(np.float32))
            fit_y.append(part["label"].to_numpy())
        eval_rows = validation & (np.array([countries.get(i) for i in ids]) == args.eval_country)
        if eval_rows.any():
            eval_frames.append(frame.filter(pl.Series(eval_rows)).select("source1_id", "target_id", *FEATURE_COLUMNS))
    fit_x, fit_y = np.vstack(fit_x), np.concatenate(fit_y)
    model = lgb.train({**PARAMS, "num_threads": args.threads}, lgb.Dataset(fit_x, fit_y, feature_name=list(FEATURE_COLUMNS)),
                      num_boost_round=args.rounds)
    del fit_x
    scored = []
    for frame in eval_frames:
        prediction = model.predict(frame.select(FEATURE_COLUMNS).to_numpy().astype(np.float32), num_threads=args.threads)
        scored.append(frame.select("source1_id", "target_id").with_columns(pl.Series("score", prediction.astype(np.float32))))
    scores = pl.concat(scored)
    top = args.top_threshold if args.top_threshold is not None else args.threshold
    result = score(to_links(decide(scores, args.threshold, top, one_owner=True)), truth)
    return {"train_country": args.train_country, "eval_country": args.eval_country, "fit_rows": int(len(fit_y)),
            "rounds": args.rounds, "threshold": args.threshold, **result}


def top_candidates(features_dir, countries, keep, limit_per_country):
    """Top-1 (by blend) candidate row per kept Source-1 entity, capped per country."""
    frames, seen = [], {}
    for frame in batches(features_dir, ["source1_id", *FEATURE_COLUMNS]):
        top = frame.sort("blend", descending=True).unique(subset="source1_id", keep="first")
        rows = [i for i in top["source1_id"].to_list() if keep(i)]
        top = top.filter(pl.col("source1_id").is_in(rows)).with_columns(
            pl.col("source1_id").map_elements(lambda i: countries.get(i, "?"), return_dtype=pl.Utf8).alias("country"))
        for country in top["country"].unique().to_list():
            part = top.filter(pl.col("country") == country)
            room = limit_per_country - seen.get(country, 0)
            if room > 0:
                frames.append(part.head(room))
                seen[country] = seen.get(country, 0) + min(room, part.height)
    return pl.concat(frames)


def adversarial(args):
    val = top_candidates(args.val_features_dir, country_of(args.val_source1), is_validation, args.per_country)
    test_countries = country_of(args.test_source1)
    test = top_candidates(args.test_features_dir, test_countries, lambda i: is_validation(i), args.per_country)
    report = {}
    for country in sorted(set(val["country"].unique()) | set(test["country"].unique())):
        a, b = val.filter(pl.col("country") == country), test.filter(pl.col("country") == country)
        if not a.height or not b.height:
            report[country] = {"note": "present in only one side", "val_rows": a.height, "test_rows": b.height}
            continue
        x = np.vstack([a.select(FEATURE_COLUMNS).to_numpy(), b.select(FEATURE_COLUMNS).to_numpy()]).astype(np.float32)
        y = np.concatenate([np.zeros(a.height), np.ones(b.height)])
        fold = np.random.RandomState(0).randint(0, 3, len(y))
        aucs, gain = [], np.zeros(len(FEATURE_COLUMNS))
        for k in range(3):
            fit = fold != k
            model = lgb.train({"objective": "binary", "learning_rate": 0.1, "num_leaves": 31, "verbose": -1,
                               "num_threads": args.threads, "seed": 7},
                              lgb.Dataset(x[fit], y[fit], feature_name=list(FEATURE_COLUMNS)), num_boost_round=100)
            aucs.append(roc_auc_score(y[~fit], model.predict(x[~fit])))
            gain += model.feature_importance("gain")
        order = np.argsort(-gain)[:10]
        report[country] = {"val_rows": a.height, "test_rows": b.height, "auc": round(float(np.mean(aucs)), 4),
                           "top_drifting_features": [(FEATURE_COLUMNS[i], round(float(gain[i] / gain.sum()), 3)) for i in order]}
    return report


def main():
    parser = argparse.ArgumentParser(description="Country-transfer and val-vs-test drift checks.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("loco")
    p.add_argument("--features-dir", required=True)
    p.add_argument("--source1", required=True, help="train_source1.tsv (country labels)")
    p.add_argument("--truth", required=True, help="val_ground_truth.tsv")
    p.add_argument("--train-country", required=True)
    p.add_argument("--eval-country", required=True)
    p.add_argument("--train-fraction", type=float, default=0.3, help="Share of that country's training entities used.")
    p.add_argument("--rounds", type=int, default=400)
    p.add_argument("--threshold", type=float, default=0.75)
    p.add_argument("--top-threshold", type=float)
    q = sub.add_parser("adversarial")
    q.add_argument("--val-features-dir", required=True)
    q.add_argument("--test-features-dir", required=True)
    q.add_argument("--val-source1", required=True, help="train_source1.tsv")
    q.add_argument("--test-source1", required=True, help="test_source1.tsv")
    q.add_argument("--per-country", type=int, default=60_000)
    for sp in (p, q):
        sp.add_argument("--threads", type=int, default=6)
        sp.add_argument("--out")
        sp.add_argument("--features", choices=("v3", "v2"), default="v3",
                        help="v2 for feature files written by the v2 pipeline")
    args = parser.parse_args()
    global FEATURE_COLUMNS
    FEATURE_COLUMNS = V2_FEATURES if args.features == "v2" else FEATURE_COLUMNS

    if args.command == "loco":
        args.train_country, args.eval_country = args.train_country.casefold(), args.eval_country.casefold()
    report = loco(args) if args.command == "loco" else adversarial(args)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
