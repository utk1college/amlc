#!/usr/bin/env python3
"""Gate 2: does the v3 candidate set improve the real macro F-0.5 over v2?

Runs on the e2e_confirm entities only, which no earlier selection or gate
has touched. Everything except the candidate set is held fixed: the v2
feature list, the LightGBM parameters, a 2-fold entity-grouped OOF split, and
the decision rule tuned by the same grid on each configuration's own OOF.
The gain is judged against its paired-bootstrap standard error over entities.

Also reports sigma_s, the per-row log-loss SD of hard and easy negatives on
the v3 OOF predictions, used by train_matcher.py's Neyman allocation if the
training rows ever exceed the row budget.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from pair_features import V2_FEATURES
from score import read_links
from split import lab_set, unit_hash
from train_matcher import HARD_COSINES, HARD_RANKS, PARAMS, best_decision, per_entity_f05

HERE = os.path.dirname(os.path.abspath(__file__))


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def features(candidates, out_dir, args):
    if not glob.glob(os.path.join(out_dir, "*.parquet")):
        subprocess.run([sys.executable, os.path.join(HERE, "pair_features.py"), "--candidates", candidates,
                        "--database", args.database, "--index-dir", args.index_dir, "--out-dir", out_dir,
                        "--ground-truth", args.truth, "--w", str(args.w)], check=True)
    return pl.concat([pl.read_parquet(p) for p in sorted(glob.glob(os.path.join(out_dir, "*.parquet")))],
                     how="diagonal_relaxed")


def oof_scores(frame):
    columns = list(V2_FEATURES)
    ids = frame["source1_id"].to_list()
    fold = np.array([unit_hash("e2e:" + i) < 0.5 for i in ids])
    stop = np.array([unit_hash("stop:" + i) < 0.05 for i in ids])
    matrix = frame.select(columns).to_numpy().astype(np.float32)
    label = frame["label"].to_numpy().astype(np.float32)
    scores = np.zeros(len(ids), np.float32)
    iterations = []
    for f in (False, True):
        fit, early = (fold != f) & ~stop, (fold != f) & stop
        train_set = lgb.Dataset(matrix[fit], label[fit], feature_name=columns)
        model = lgb.train(PARAMS, train_set, num_boost_round=3000,
                          valid_sets=[lgb.Dataset(matrix[early], label[early], reference=train_set)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        scores[fold == f] = model.predict(matrix[fold == f])
        iterations.append(model.best_iteration)
    return frame.select("source1_id", "target_id", "label").with_columns(pl.Series("score", scores)), iterations


def stratum_sigma(frame, scored):
    """Per-row log-loss SD of hard and easy negatives on OOF predictions."""
    cosines = [c for c in HARD_COSINES if c in frame.columns and frame[c].is_not_null().any()]
    ranks = [c for c in HARD_RANKS if c in frame.columns and frame[c].is_not_null().any()]
    positives = frame.filter(pl.col("label") == 1)
    hard = np.zeros(frame.height, bool)
    for c in cosines + ranks:
        values = positives[c].drop_nulls().drop_nans() if frame[c].dtype.is_float() else positives[c].drop_nulls()
        if not len(values):
            continue
        cut = float(np.quantile(values.to_numpy(), 0.05 if c in cosines else 0.95))
        column = frame[c].to_numpy()
        hard |= np.isfinite(column) & ((column >= cut) if c in cosines else (column <= cut))
    p = np.clip(scored["score"].to_numpy(), 1e-7, 1 - 1e-7)
    y = scored["label"].to_numpy()
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    negative = y == 0
    return {"hard": float(loss[negative & hard].std()), "easy": float(loss[negative & ~hard].std()),
            "hard_rows": int((negative & hard).sum()), "easy_rows": int((negative & ~hard).sum())}


def main():
    parser = argparse.ArgumentParser(description="End-to-end gate on e2e_confirm entities.")
    parser.add_argument("--v2-candidates", required=True, help="e2e_v2.tsv from run_lab.sh export")
    parser.add_argument("--v3-candidates", required=True, help="e2e_v3 Parquet dir (or TSV) from run_lab.sh export")
    parser.add_argument("--database", required=True, help="v3 record store for the train split")
    parser.add_argument("--index-dir", required=True, help="v3 forward index for the train split (IDF)")
    parser.add_argument("--truth", required=True)
    parser.add_argument("--w", type=float, required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    started = time.time()
    truth = {s: t for s, t in read_links(args.truth).items() if lab_set(s) == "e2e_confirm"}
    report, per_entity = {"entities": len(truth)}, {}
    frames = {}
    for name, candidates in (("v2", args.v2_candidates), ("v3", args.v3_candidates)):
        frames[name] = features(candidates, os.path.join(args.work_dir, f"e2e_features_{name}"), args)
        scored, iterations = oof_scores(frames[name])
        decision = best_decision(scored, truth)
        ids, values = per_entity_f05(scored, truth, decision)
        per_entity[name] = values
        report[name] = {"pairs": frames[name].height, "best_iterations": iterations, **decision}
        log(f"{name}: macro F0.5 {decision['macro_f05']}")
        if name == "v3":
            report["sigma"] = stratum_sigma(frames[name], scored)
    diff = per_entity["v3"] - per_entity["v2"]
    boot = np.random.default_rng(0).integers(0, len(diff), (200, len(diff)))
    se = float(diff[boot].mean(axis=1).std())
    report["gate2"] = {"gain": float(diff.mean()), "se": se, "pass": bool(diff.mean() > 2 * se)}
    report["seconds"] = round(time.time() - started, 1)
    text = json.dumps(report, indent=2)
    print(text)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
