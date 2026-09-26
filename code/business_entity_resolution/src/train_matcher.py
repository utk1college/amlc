#!/usr/bin/env python3
"""Stage-1 pair model: LightGBM with 3 entity-grouped out-of-fold folds.

Every non-validation Source-1 entity belongs to one fold (split.oof_fold).
The model for fold f trains on the other folds (with 5% of their entities
held out for early stopping) and scores every candidate pair of fold f, so
each training pair gets an out-of-fold (OOF) score from a model that never
saw its entity. Validation pairs are scored by the mean of the fold models,
exactly like test pairs (predict.py), so all three score sets share one
distribution. The 10% validation entities are never used for fitting.

Rows are streamed from the feature Parquet files through lgb.Sequence, so the
float matrix is never held in memory; LightGBM keeps only its binned copy.

Sampling (only if the rows exceed --row-budget; by default everything is used):
positives are always kept; negatives are split into a hard stratum (some
view cosine >= that view's 5th percentile among positives, or some pass rank
<= the 95th percentile of positives' ranks, both measured on the fold's own
training entities) and an easy stratum. The budget left after positives is
split by Neyman allocation n_s ~ N_s * sigma_s (sigma_s: per-row log-loss SD
of the stratum, from the end-to-end gate), capped at N_s, and each sampled row
is weighted N_s / n_s so the training loss stays unbiased.
"""

import argparse
import glob
import json
import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from decide import decide, to_links
from pair_features import FEATURE_COLUMNS, V2_FEATURES
from score import read_links, score
from split import is_validation, oof_fold, unit_hash

PARAMS = {
    "objective": "binary", "learning_rate": 0.05, "num_leaves": 255, "min_data_in_leaf": 200,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
    "max_bin": 255, "num_threads": int(os.environ.get("WORKERS", os.cpu_count() or 4)), "verbose": -1, "seed": 7,
}
FOLDS = 3
HARD_COSINES = ("cos_name", "cos_address", "cos_address_char", "cos_c1", "emb_score")
HARD_RANKS = ("rank_name", "rank_address", "rank_address_char", "rank_c1", "rev_rank_name",
              "rev_rank_address", "rev_rank_c1", "emb_rank")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def feature_list(name):
    return list(V2_FEATURES if name == "v2" else FEATURE_COLUMNS)


class ParquetRows(lgb.Sequence):
    """Selected rows of a list of feature Parquet files, loaded one file at a time."""

    def __init__(self, files, selections, columns, batch_size=65_536):
        self.files, self.selections, self.columns = files, selections, columns
        self.offsets = np.cumsum([0] + [len(s) for s in selections])
        self.batch_size = batch_size
        self.cached, self.matrix = None, None

    def __len__(self):
        return int(self.offsets[-1])

    def rows_of(self, i):
        if self.cached != i:
            frame = pl.read_parquet(self.files[i], columns=self.columns)
            # LightGBM's Sequence sampling requires float64 rows.
            self.matrix = frame.to_numpy().astype(np.float64, copy=False)[self.selections[i]]
            self.cached = i
        return self.matrix

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            f = int(np.searchsorted(self.offsets, index, side="right") - 1)
            return self.rows_of(f)[index - self.offsets[f]]
        if isinstance(index, slice):
            index = np.arange(*index.indices(len(self)))
        index = np.asarray(index)
        out = np.empty((len(index), len(self.columns)), np.float64)
        owners = np.searchsorted(self.offsets, index, side="right") - 1
        for f in np.unique(owners):
            mask = owners == f
            out[mask] = self.rows_of(int(f))[index[mask] - self.offsets[f]]
        return out


class RowIndex:
    """Per-file labels, folds and hardness inputs for the non-validation rows."""

    def __init__(self, files, shuffle_seed=None):
        self.files, self.rows, self.labels, self.folds, self.stop, self.curve = files, [], [], [], [], []
        cache = {}
        for path in files:
            frame = pl.read_parquet(path, columns=["source1_id", "is_validation", "label"])
            keep = np.flatnonzero(~frame["is_validation"].to_numpy())
            ids = frame["source1_id"].to_numpy()[keep]
            folds, stop, curve = np.empty(len(ids), np.int8), np.empty(len(ids), bool), np.empty(len(ids), np.float32)
            for j, source1_id in enumerate(ids):
                cached = cache.get(source1_id)
                if cached is None:
                    cached = cache[source1_id] = (oof_fold(source1_id), unit_hash("stop:" + source1_id) < 0.05,
                                                  unit_hash("curve:" + source1_id))
                folds[j], stop[j], curve[j] = cached
            self.rows.append(keep)
            self.labels.append(frame["label"].to_numpy()[keep].astype(np.float32))
            self.folds.append(folds)
            self.stop.append(stop)
            self.curve.append(curve)
        if shuffle_seed is not None:
            rng = np.random.default_rng(shuffle_seed)
            everything = rng.permutation(np.concatenate(self.labels))
            cuts = np.cumsum([len(x) for x in self.labels])[:-1]
            self.labels = np.split(everything, cuts)
        self.total = sum(len(x) for x in self.rows)

    def hardness(self, train_mask_per_file):
        """Hard-negative flags per file, with cut-offs from the positives of the masked rows."""
        present = pl.read_parquet(self.files[0], n_rows=1).columns
        cosines = [c for c in HARD_COSINES if c in present]
        ranks = [c for c in HARD_RANKS if c in present]
        positives = {c: [] for c in cosines + ranks}
        for path, rows, labels, mask in zip(self.files, self.rows, self.labels, train_mask_per_file):
            frame = pl.read_parquet(path, columns=cosines + ranks)[rows]
            chosen = mask & (labels == 1)
            for c in cosines + ranks:
                values = frame[c].to_numpy()[chosen]
                positives[c].append(values[np.isfinite(values)])
        cut = {c: float(np.quantile(np.concatenate(positives[c]), 0.05 if c in cosines else 0.95))
               for c in positives if sum(len(v) for v in positives[c])}
        flags = []
        for path, rows in zip(self.files, self.rows):
            frame = pl.read_parquet(path, columns=list(cut))[rows]
            hard = np.zeros(frame.height, bool)
            for c, value in cut.items():
                column = frame[c].to_numpy()
                hard |= np.isfinite(column) & ((column >= value) if c in cosines else (column <= value))
            flags.append(hard)
        return flags, cut


def neyman_rates(n_positive, n_hard, n_easy, budget, sigma):
    """Inclusion rates (hard, easy) for a row budget; all positives are kept."""
    room = budget - n_positive
    if room >= n_hard + n_easy:
        return 1.0, 1.0
    weights = {"hard": n_hard * sigma["hard"], "easy": n_easy * sigma["easy"]}
    sizes = {"hard": n_hard, "easy": n_easy}
    allocation, remaining, open_strata = {}, room, ["hard", "easy"]
    while open_strata:
        total = sum(weights[s] for s in open_strata) or 1.0
        capped = [s for s in open_strata if remaining * weights[s] / total >= sizes[s]]
        if not capped:
            for s in open_strata:
                allocation[s] = remaining * weights[s] / total
            break
        for s in capped:
            allocation[s] = sizes[s]
            remaining -= sizes[s]
            open_strata.remove(s)
    return (min(1.0, allocation["hard"] / max(n_hard, 1)), min(1.0, allocation["easy"] / max(n_easy, 1)))


def training_rows(index, fold, args, exclude_fraction=None):
    """Per-file selections and weights for the model of `fold` (train rows + early-stop rows)."""
    train_masks = [(f != fold) for f in index.folds]
    if exclude_fraction is not None:  # learning curve: drop a share of the training *entities*
        train_masks = [m & (c >= exclude_fraction) for m, c in zip(train_masks, index.curve)]
    n_train = sum(int(m.sum()) for m in train_masks)
    rates, cut, strata = (1.0, 1.0), None, None
    if args.row_budget and n_train > args.row_budget:
        hard, cut = index.hardness(train_masks)
        n_pos = sum(int((m & (y == 1)).sum()) for m, y in zip(train_masks, index.labels))
        n_hard = sum(int((m & (y == 0) & h).sum()) for m, y, h in zip(train_masks, index.labels, hard))
        n_easy = n_train - n_pos - n_hard
        rates = neyman_rates(n_pos, n_hard, n_easy, args.row_budget,
                             {"hard": args.sigma_hard, "easy": args.sigma_easy})
        strata = {"positives": n_pos, "hard": n_hard, "easy": n_easy, "rates": rates, "cut_offs": cut}
    fit, stop = ([], [], []), ([], [], [])
    for i, (rows, labels, folds, stops, mask) in enumerate(zip(index.rows, index.labels, index.folds, index.stop, train_masks)):
        weight = np.ones(len(rows), np.float32)
        keep = mask.copy()
        if rates != (1.0, 1.0):
            draw = np.random.default_rng(1000 * (fold + 1) + i).random(len(rows))  # deterministic per file
            negative = labels == 0
            for is_hard, rate in ((hard[i], rates[0]), (~hard[i], rates[1])):
                stratum = mask & negative & is_hard
                keep &= ~stratum | (draw < rate)
                weight[stratum] = 1.0 / rate
        for target, chosen in ((fit, keep & ~stops), (stop, keep & stops)):
            target[0].append(rows[chosen])
            target[1].append(labels[chosen])
            target[2].append(weight[chosen])
    return fit, stop, strata


def train_one(index, fold, columns, args, exclude_fraction=None, rounds=None):
    fit, stop, strata = training_rows(index, fold, args, exclude_fraction)
    train_set = lgb.Dataset([ParquetRows(index.files, fit[0], columns)], label=np.concatenate(fit[1]),
                            weight=np.concatenate(fit[2]), feature_name=columns, free_raw_data=True,
                            params={"max_bin": PARAMS["max_bin"]})
    stop_set = lgb.Dataset([ParquetRows(index.files, stop[0], columns)], label=np.concatenate(stop[1]),
                           weight=np.concatenate(stop[2]), reference=train_set)
    log(f"fold {fold}: {sum(len(x) for x in fit[0])} fit rows, {sum(len(x) for x in stop[0])} early-stop rows")
    model = lgb.train(PARAMS, train_set, num_boost_round=rounds or args.rounds, valid_sets=[stop_set],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    return model, strata


def score_rows(files, models, index, columns, out_path, only_fold=None):
    """OOF scores (each fold's rows by its model) and validation scores (mean of models)."""
    frames = []
    for path, rows, folds in zip(files, index.rows, index.folds):
        frame = pl.read_parquet(path, columns=["source1_id", "target_id", "is_validation", "label", *columns])
        matrix = frame.select(columns).to_numpy().astype(np.float32)
        scores = np.full(frame.height, np.nan, np.float32)
        fold_of_row = np.full(frame.height, -1, np.int8)
        fold_of_row[rows] = folds
        for f, model in models.items():
            mask = fold_of_row == f
            if mask.any():
                scores[mask] = model.predict(matrix[mask])
        validation = frame["is_validation"].to_numpy()
        if only_fold is None and validation.any():
            scores[validation] = np.mean([m.predict(matrix[validation]) for m in models.values()], axis=0)
        keep = np.isfinite(scores)
        frames.append(frame.select("source1_id", "target_id", "is_validation", "label").filter(pl.Series(keep))
                      .with_columns(pl.Series("score", scores[keep])))
    table = pl.concat(frames)
    table.write_parquet(out_path)
    return table


def best_decision(scores, truth):
    """Best (one-owner, threshold, top-threshold) on these scores by exact macro F0.5."""
    scores = scores.filter(pl.col("score") >= 0.05).select("source1_id", "target_id", "score")
    best = None
    for one_owner in (True, False):
        for step in range(8, 19):
            threshold = step / 20
            for drop in (0.0, 0.1, 0.2):
                top = round(max(threshold - drop, 0.05), 2)
                result = score(to_links(decide(scores, threshold, top, one_owner)), truth)
                if best is None or result["macro_f05"] > best["macro_f05"]:
                    best = {"one_owner": one_owner, "threshold": threshold, "top_threshold": top, **result}
    return best


def per_entity_f05(scores, truth, decision):
    from score import entity_f05
    links = to_links(decide(scores.filter(pl.col("score") >= 0.05).select("source1_id", "target_id", "score"),
                            decision["threshold"], decision["top_threshold"], decision["one_owner"]))
    ids = sorted(truth)
    return ids, np.array([entity_f05(links.get(s, set()), truth[s]) for s in ids])


def main():
    parser = argparse.ArgumentParser(description="3-fold OOF LightGBM pair model.")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--truth", required=True, help="train_ground_truth.tsv")
    parser.add_argument("--model-dir", required=True, help="Writes fold models, OOF/validation scores and report")
    parser.add_argument("--features", choices=("v3", "v2"), default="v3")
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--row-budget", type=int, default=0, help="Max training rows (0 = use every row)")
    parser.add_argument("--sigma-hard", type=float, default=1.0, help="Per-row log-loss SD, hard stratum (gate 2)")
    parser.add_argument("--sigma-easy", type=float, default=1.0, help="Per-row log-loss SD, easy stratum (gate 2)")
    parser.add_argument("--shuffle-labels", action="store_true", help="Leak check: fold 0 only, permuted labels")
    parser.add_argument("--learning-curve", action="store_true", help="Also fit fold 0 on half its training entities")
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.model_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.features_dir, "*.parquet")))
    columns = feature_list(args.features)
    index = RowIndex(files, shuffle_seed=0 if args.shuffle_labels else None)
    truth_all = read_links(args.truth)
    report = {"features": args.features, "columns": len(columns), "rows": index.total}

    if args.shuffle_labels:
        model, _ = train_one(index, 0, columns, args, rounds=100)
        oof = score_rows(files, {0: model}, index, columns, os.path.join(args.model_dir, "shuffled_oof.parquet"), 0)
        entities = set(oof["source1_id"].unique().to_list())
        report["shuffled_labels_fold0"] = best_decision(oof, {s: t for s, t in truth_all.items() if s in entities
                                                              and oof_fold(s) == 0})
        print(json.dumps(report, indent=2))
        return

    models, strata = {}, {}
    for fold in range(FOLDS):
        models[fold], strata[fold] = train_one(index, fold, columns, args)
        models[fold].save_model(os.path.join(args.model_dir, f"fold{fold}.txt"))
        report.setdefault("best_iteration", {})[fold] = models[fold].best_iteration
    report["sampling"] = strata
    table = score_rows(files, models, index, columns, os.path.join(args.model_dir, "scores.parquet"))
    oof = table.filter(~pl.col("is_validation"))
    truth_train = {s: t for s, t in truth_all.items() if not is_validation(s)}
    report["oof"] = best_decision(oof, truth_train)
    countries = {}
    source1 = os.path.join(os.path.dirname(args.truth), "train_source1.tsv")
    if os.path.exists(source1):
        with open(source1, encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                countries[parts[0]] = parts[3].strip().casefold()
        ids, values = per_entity_f05(oof, truth_train, report["oof"])
        by_country = {}
        for source1_id, value in zip(ids, values):
            by_country.setdefault(countries.get(source1_id, "?"), []).append(value)
        report["oof_by_country"] = {c: {"entities": len(v), "macro_f05": round(float(np.mean(v)), 6)}
                                    for c, v in by_country.items()}
    if args.learning_curve:
        half, _ = train_one(index, 0, columns, args, exclude_fraction=0.5)
        fold0 = {s: t for s, t in truth_train.items() if oof_fold(s) == 0}
        half_scores = score_rows(files, {0: half}, index, columns, os.path.join(args.model_dir, "lc_half.parquet"), 0)
        full_scores = oof.filter(pl.col("source1_id").is_in(list(fold0)))
        a_ids, a = per_entity_f05(half_scores, fold0, best_decision(half_scores, fold0))
        b_ids, b = per_entity_f05(full_scores, fold0, best_decision(full_scores, fold0))
        diff = b - a
        boot = np.random.default_rng(0).integers(0, len(diff), (200, len(diff)))
        report["learning_curve_fold0"] = {"half_training_entities": float(a.mean()), "all_training_entities": float(b.mean()),
                                          "gain": float(diff.mean()), "se": float(diff[boot].mean(axis=1).std()),
                                          "data_limited": bool(diff.mean() > 2 * diff[boot].mean(axis=1).std())}
    importance = sorted(zip(columns, models[0].feature_importance("gain")), key=lambda x: -x[1])
    report["top_features_by_gain_fold0"] = [(n, round(float(g), 1)) for n, g in importance[:25]]
    report["seconds"] = round(time.time() - started, 1)
    with open(os.path.join(args.model_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
