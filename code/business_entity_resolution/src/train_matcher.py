#!/usr/bin/env python3
"""Train the LightGBM pair classifier, then score validation pairs batch by batch.

Training rows are the non-validation pairs in the feature directory; 5% of
their Source-1 entities are held out for early stopping. Validation pairs
are never used for fitting; they are streamed through the model one feature
batch at a time and written to a single scores file for decide.py.
"""

import argparse
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pair_features import FEATURE_COLUMNS
from split import is_validation

PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "max_bin": 255, "num_threads": 6, "verbose": -1, "seed": 7,
}


def read_batch(path, columns):
    return pq.read_table(path, columns=columns)


def matrix(table):
    return np.column_stack([table.column(c).to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)


def load_training(paths):
    fit_x, fit_y, stop_x, stop_y = [], [], [], []
    for path in paths:
        table = read_batch(path, ["source1_id", "is_validation", "label", *FEATURE_COLUMNS])
        train_rows = ~table.column("is_validation").to_numpy(zero_copy_only=False)
        ids = table.column("source1_id").to_pylist()
        stop_rows = np.array([is_validation("stop" + s, 0.05) for s in ids]) & train_rows
        x, y = matrix(table), table.column("label").to_numpy()
        fit_x.append(x[train_rows & ~stop_rows])
        fit_y.append(y[train_rows & ~stop_rows])
        stop_x.append(x[stop_rows])
        stop_y.append(y[stop_rows])
    return np.vstack(fit_x), np.concatenate(fit_y), np.vstack(stop_x), np.concatenate(stop_y)


def main():
    parser = argparse.ArgumentParser(description="Train the pair classifier and score validation pairs.")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--model-out", required=True)
    parser.add_argument("--scores-out", required=True, help="Validation pair scores (Parquet).")
    parser.add_argument("--rounds", type=int, default=2000)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.features_dir, "*.parquet")))
    started = time.time()
    fit_x, fit_y, stop_x, stop_y = load_training(paths)
    print(f"fit rows {len(fit_y)} (positives {fit_y.mean():.3f}), early-stop rows {len(stop_y)}", flush=True)
    fit = lgb.Dataset(fit_x, fit_y, feature_name=list(FEATURE_COLUMNS), free_raw_data=True)
    stop = lgb.Dataset(stop_x, stop_y, reference=fit)
    del fit_x, stop_x
    model = lgb.train(PARAMS, fit, num_boost_round=args.rounds, valid_sets=[stop],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
    model.save_model(args.model_out)
    importance = sorted(zip(FEATURE_COLUMNS, model.feature_importance("gain")), key=lambda x: -x[1])

    writer, scored = None, 0
    for path in paths:
        table = read_batch(path, ["source1_id", "target_id", "is_validation", "label", *FEATURE_COLUMNS])
        rows = table.column("is_validation").to_numpy(zero_copy_only=False)
        if not rows.any():
            continue
        table = table.filter(pa.array(rows))
        out = pa.table({"source1_id": table.column("source1_id"), "target_id": table.column("target_id"),
                        "score": model.predict(matrix(table), num_threads=6).astype(np.float32),
                        "label": table.column("label")})
        writer = writer or pq.ParquetWriter(args.scores_out, out.schema)
        writer.write_table(out)
        scored += out.num_rows
    if writer:
        writer.close()
    report = {"best_iteration": model.best_iteration, "validation_pairs_scored": scored,
              "seconds": round(time.time() - started, 1),
              "top_features_by_gain": [(n, round(float(g), 1)) for n, g in importance[:15]]}
    with open(args.model_out + ".json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
