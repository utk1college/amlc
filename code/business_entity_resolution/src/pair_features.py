#!/usr/bin/env python3
"""Pairwise features for candidate pairs, computed batch by batch in a process pool.

Candidates come from a candidate TSV (lists only) or from a merge_candidates
Parquet directory (lists plus retrieval metadata). Each batch of Source-1
entities fetches exactly its records from the record store, computes its
features and writes one Parquet file. Every feature is label-free; with
--ground-truth a `label` column is added from the train labels.

Feature groups
  V2_FEATURES   the 42 features of the v2 matcher, unchanged (the end-to-end
                gate compares candidate sets with these held fixed)
  FEATURE_COLUMNS (v3) = V2 minus target_non_ascii, plus
    - script flags (non-Latin / accented) and the embedding trigger
    - alias-aware name similarity, core-token containment, skeleton
      Levenshtein / LCS / prefix similarity
    - postal agreement (5-6 digit tokens) kept apart from house numbers
    - exact TF-IDF cosines per view (name, address, address_char, C1) and
      each one's ratio to the entity's best
    - per-source rank and gap of the blend score within the entity
    - retrieval metadata: censored per-pass ranks, reverse rank/score, the
      target's best and runner-up reverse score, embedding rank/score
"""

import argparse
import glob
import math
import os
import sqlite3
import sys
import time
from multiprocessing import get_all_start_methods, get_context

import numpy as np
import polars as pl
import scipy.sparse as sp
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, LCSseq, Levenshtein, Prefix
from rapidfuzz.process import cpdist
from sklearn.utils import murmurhash3_32

from blocking_knn import VECTORIZERS, tfidf
from record_store import COLUMNS, RecordStore, bounded_imap
from split import is_validation
from text_normalize import LEGAL_WORDS

N_FEATURES = 2 ** 22  # must match blocking_knn.N_FEATURES
F = {name: i for i, name in enumerate(COLUMNS)}
LEGAL_FORMS_ONLY = LEGAL_WORDS - {"the", "and"}
CONTEXT = get_context("fork" if "fork" in get_all_start_methods() else "spawn")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


class IdfTable:
    """IDF of name skeleton words and address words, per country, from the blocking index."""

    def __init__(self, index_dir):
        self.index_dir, self.tables, self.cache = index_dir, {}, {}

    def table(self, country):
        if country not in self.tables:
            path = os.path.join(self.index_dir, country.replace(" ", "_") + ".idf.npy")
            self.tables[country] = np.load(path) if os.path.exists(path) else None
        return self.tables[country]

    def weight(self, country, view, token):
        key = (country, view, token)
        value = self.cache.get(key)
        if value is None:
            table = self.table(country)
            feature = ("w" + token) if view == 0 else token
            value = float(table[view][abs(murmurhash3_32(feature, seed=0)) % N_FEATURES]) if table is not None else 1.0
            self.cache[key] = value
        return value


def idf_overlap(a_tokens, b_tokens, weight):
    a, b = set(a_tokens), set(b_tokens)
    if not a or not b:
        return 0.0, 0.0, 0.0, 0.0
    shared = sum(weight(t) for t in a & b)
    union = shared + sum(weight(t) for t in a ^ b)
    missing = max((weight(t) for t in a - b), default=0.0)
    extra = max((weight(t) for t in b - a), default=0.0)
    return shared / union, shared, missing, extra


def numbers(tokens):
    return [t for t in tokens if t.isdigit()]


def agreement(a, b):
    """1 if the sets share an element, 0 if both are non-empty and disjoint, NaN otherwise."""
    if not a or not b:
        return math.nan
    return float(bool(a & b))


def python_features(a, b, idf):
    """Per-pair features that need token logic (a, b are record-store rows)."""
    country = a[F["country"]]
    a_words, b_words = a[F["name_skeleton"]].split(), b[F["name_skeleton"]].split()
    a_core, b_core = a[F["core"]].split(), b[F["core"]].split()
    a_legal = {w for w in a[F["name"]].split() if w in LEGAL_FORMS_ONLY}
    b_legal = {w for w in b[F["name"]].split() if w in LEGAL_FORMS_ONLY}
    name_jaccard, name_shared, name_missing, name_extra = idf_overlap(
        a_words, b_words, lambda t: idf.weight(country, 0, t))
    a_address, b_address = a[F["address_skeleton"]].split(), b[F["address_skeleton"]].split()
    address_jaccard, address_shared, address_missing, address_extra = idf_overlap(
        a_address, b_address, lambda t: idf.weight(country, 1, t))
    a_numbers, b_numbers = numbers(a[F["address"]].split()), numbers(b[F["address"]].split())
    number_union = set(a_numbers) | set(b_numbers)
    first_gap = (math.log1p(abs(int(a_numbers[0][:9]) - int(b_numbers[0][:9])))
                 if a_numbers and b_numbers else -1.0)
    a_postal = {n for n in a_numbers if 5 <= len(n) <= 6}
    b_postal = {n for n in b_numbers if 5 <= len(n) <= 6}
    a_house = [n for n in a_numbers if len(n) <= 4]
    b_house = [n for n in b_numbers if len(n) <= 4]
    shared_core = len(set(a_core) & set(b_core))
    v2 = (
        name_jaccard, name_shared, name_missing, name_extra,
        float(" ".join(a_core) == " ".join(b_core)),
        float(bool(a_core) and bool(b_core) and a_core[0] == b_core[0]),
        len(a_words), len(b_words),
        float(bool(a_legal)), float(bool(b_legal)),
        float(bool(a_legal) and bool(b_legal) and not (a_legal & b_legal)),
        address_jaccard, address_shared, address_missing, address_extra,
        float(not a[F["address"]]), float(not b[F["address"]]),
        len(set(a_numbers) & set(b_numbers)) / len(number_union) if number_union else -1.0,
        float(bool(a_numbers) and bool(b_numbers) and a_numbers[0] == b_numbers[0]),
        first_gap,
        float(b[F["nonlatin"]] or b[F["accented"]]), float(b[F["id"]].startswith("S3")),
    )
    v3 = (
        float(b[F["nonlatin"]]), float(b[F["accented"]]), float(b[F["nonlatin"]]),
        float(bool(b[F["variants"]])),
        shared_core / min(len(set(a_core)), len(set(b_core))) if a_core and b_core else math.nan,
        agreement(a_postal, b_postal),
        agreement({a_house[0]} if a_house else set(), {b_house[0]} if b_house else set()),
    )
    return v2 + v3


V2_PYTHON = (
    "name_idf_jaccard", "name_idf_shared", "name_idf_missing_max", "name_idf_extra_max",
    "core_equal", "first_core_equal", "name_words_s1", "name_words_target",
    "legal_s1", "legal_target", "legal_conflict",
    "address_idf_jaccard", "address_idf_shared", "address_idf_missing_max", "address_idf_extra_max",
    "address_missing_s1", "address_missing_target",
    "number_jaccard", "first_number_equal", "first_number_log_gap",
    "target_non_ascii", "target_source3",
)
V3_PYTHON = ("target_nonlatin_script", "target_accented", "embedding_triggered", "has_alias",
             "core_containment", "postal_equal", "first_house_number_equal")
PYTHON_COLUMNS = V2_PYTHON + V3_PYTHON
V2_FUZZY = {
    # column: (scorer, record field)
    "name_ratio": (fuzz.ratio, "name"), "name_partial": (fuzz.partial_ratio, "name"),
    "name_token_sort": (fuzz.token_sort_ratio, "name"), "name_token_set": (fuzz.token_set_ratio, "name"),
    "name_jaro_winkler": (JaroWinkler.normalized_similarity, "name"),
    "skeleton_ratio": (fuzz.ratio, "name_skeleton"), "skeleton_token_set": (fuzz.token_set_ratio, "name_skeleton"),
    "address_ratio": (fuzz.ratio, "address"), "address_partial": (fuzz.partial_ratio, "address"),
    "address_token_set": (fuzz.token_set_ratio, "address"),
    "address_skeleton_token_set": (fuzz.token_set_ratio, "address_skeleton"),
}
V3_FUZZY = {
    "skeleton_levenshtein": (Levenshtein.normalized_similarity, "name_skeleton"),
    "skeleton_lcs": (LCSseq.normalized_similarity, "name_skeleton"),
    "core_prefix": (Prefix.normalized_similarity, "core"),
}
GROUP_SOURCES = ("name_token_set", "address_token_set", "blend")
V2_FEATURES = (tuple(V2_FUZZY) + V2_PYTHON + ("blend",)
               + tuple(f"{c}_rank" for c in GROUP_SOURCES)
               + tuple(f"{c}_gap_to_best" for c in GROUP_SOURCES)
               + ("group_size", "group_strong_names"))
COSINES = ("cos_name", "cos_address", "cos_address_char", "cos_c1")
META_COLUMNS = ("rank_name", "rank_address", "rank_address_char", "rank_c1",
                "rev_rank_name", "rev_rank_address", "rev_rank_c1",
                "rev_score_name", "rev_score_address", "rev_score_c1",
                "rev_best_c1", "rev_second_c1", "rev_gap_c1", "rev_margin_c1",
                "emb_rank", "emb_score", "n_passes")
FEATURE_COLUMNS = (tuple(c for c in V2_FEATURES if c != "target_non_ascii") + V3_PYTHON + tuple(V3_FUZZY)
                   + ("alias_token_set", "alias_ratio") + COSINES + tuple(f"{c}_ratio_to_best" for c in COSINES)
                   + ("blend_source_rank", "blend_source_gap_to_best") + META_COLUMNS)
ALL_COLUMNS = tuple(dict.fromkeys(V2_FEATURES + FEATURE_COLUMNS))


def group_features(columns, offsets, sources):
    """Rank and gap-to-best of each pair within its Source-1 entity, plus per-source blend rank."""
    n = len(columns["blend"])
    out = {f"{c}_rank": np.zeros(n, np.float32) for c in GROUP_SOURCES}
    out.update({f"{c}_gap_to_best": np.zeros(n, np.float32) for c in GROUP_SOURCES})
    out.update({name: np.zeros(n, np.float32) for name in ("group_size", "group_strong_names",
                                                           "blend_source_rank", "blend_source_gap_to_best")})
    out.update({f"{c}_ratio_to_best": np.full(n, np.nan, np.float32) for c in COSINES})
    for start, end in zip(offsets[:-1], offsets[1:]):
        if start == end:
            continue
        out["group_size"][start:end] = end - start
        out["group_strong_names"][start:end] = np.sum(columns["name_token_set"][start:end] >= 90)
        for c in GROUP_SOURCES:
            values = columns[c][start:end]
            order = np.argsort(-values, kind="stable")
            ranks = np.empty(end - start, np.float32)
            ranks[order] = np.arange(1, end - start + 1)
            out[f"{c}_rank"][start:end] = ranks
            out[f"{c}_gap_to_best"][start:end] = values.max() - values
        for c in COSINES:
            values = columns[c][start:end]
            best = np.nanmax(values) if np.isfinite(values).any() else 0.0
            if best > 0:
                out[f"{c}_ratio_to_best"][start:end] = values / best
        blend, source = columns["blend"][start:end], sources[start:end]
        for s in np.unique(source):
            mask = source == s
            values = blend[mask]
            order = np.argsort(-values, kind="stable")
            ranks = np.empty(len(values), np.float32)
            ranks[order] = np.arange(1, len(values) + 1)
            out["blend_source_rank"][start:end][mask] = ranks
            out["blend_source_gap_to_best"][start:end][mask] = values.max() - values
    return out


def exact_cosines(pairs, records, index_dir, w, idf_cache):
    """Exact TF-IDF cosines per view, with the same vectorizers and IDF as blocking."""
    out = {c: np.full(len(pairs), np.nan, np.float32) for c in COSINES}
    by_country = {}
    for i, (s, _) in enumerate(pairs):
        by_country.setdefault(records[s][F["country"]], []).append(i)
    for country, rows in by_country.items():
        if country not in idf_cache:
            path = os.path.join(index_dir, country.replace(" ", "_") + ".idf.npy")
            idf_cache[country] = np.load(path) if os.path.exists(path) else None
        idf = idf_cache[country]
        if idf is None:
            continue
        ids = list(dict.fromkeys([pairs[i][0] for i in rows] + [pairs[i][1] for i in rows]))
        position = {x: j for j, x in enumerate(ids)}
        texts = {"name": [records[x][F["name"]] for x in ids], "address": [records[x][F["address"]] for x in ids]}
        texts["address_char"] = texts["address"]
        a_rows = np.array([position[pairs[i][0]] for i in rows])
        b_rows = np.array([position[pairs[i][1]] for i in rows])
        for j, view in enumerate(("name", "address", "address_char")):
            if j >= len(idf):
                continue
            matrix = tfidf(VECTORIZERS[view].transform(texts[view]), idf[j])
            value = np.asarray(matrix[a_rows].multiply(matrix[b_rows]).sum(axis=1)).ravel()
            out[f"cos_{view}"][rows] = value
        out["cos_c1"][rows] = w * out["cos_name"][rows] + (1 - w) * out["cos_address"][rows]
    return out


def batch_features(pairs, records, idf, index_dir, w, idf_cache):
    """pairs: list of (source1_id, target_id), grouped by source1_id."""
    a_rows = [records[s] for s, _ in pairs]
    b_rows = [records[t] for _, t in pairs]
    columns = {}
    for name, (scorer, field) in {**V2_FUZZY, **V3_FUZZY}.items():
        columns[name] = cpdist([r[F[field]] for r in a_rows], [r[F[field]] for r in b_rows],
                               scorer=scorer, workers=1).astype(np.float32)
    for name in ("name_jaro_winkler", "skeleton_levenshtein", "skeleton_lcs", "core_prefix"):
        columns[name] *= 100
    python = np.array([python_features(a, b, idf) for a, b in zip(a_rows, b_rows)], dtype=np.float32)
    for i, name in enumerate(PYTHON_COLUMNS):
        columns[name] = python[:, i]
    columns["blend"] = (0.6 * columns["name_token_set"] + 0.4 * columns["address_token_set"]).astype(np.float32)
    columns["alias_token_set"] = columns["name_token_set"].copy()
    columns["alias_ratio"] = columns["name_ratio"].copy()
    for i, (a, b) in enumerate(zip(a_rows, b_rows)):
        if b[F["variants"]]:
            for variant in b[F["variants"]].split("|"):
                columns["alias_token_set"][i] = max(columns["alias_token_set"][i], fuzz.token_set_ratio(a[F["name"]], variant))
                columns["alias_ratio"][i] = max(columns["alias_ratio"][i], fuzz.ratio(a[F["name"]], variant))
    columns.update(exact_cosines(pairs, records, index_dir, w, idf_cache))
    return columns


def generic_metadata(frame):
    """Rename config-specific metadata columns to the fixed META_COLUMNS names."""
    renames = {}
    for column in frame.columns:
        for prefix in ("rank_combined_", "rev_rank_combined_", "rev_score_combined_", "rev_best_combined_",
                       "rev_second_combined_"):
            if column.startswith(prefix):
                renames[column] = prefix.replace("combined_", "c1")
    frame = frame.rename(renames)
    extra = []
    if "rev_best_c1" in frame.columns and "rev_score_c1" in frame.columns:
        extra.append((pl.col("rev_best_c1") - pl.col("rev_score_c1")).alias("rev_gap_c1"))
    if "rev_best_c1" in frame.columns and "rev_second_c1" in frame.columns:
        extra.append((pl.col("rev_best_c1") - pl.col("rev_second_c1")).alias("rev_margin_c1"))
    return frame.with_columns(extra) if extra else frame


# ---------------------------------------------------------------- batching

def read_tsv_batches(path, size):
    batch = []
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            source1_id, _, ids = line.rstrip("\n").partition("\t")
            batch.append((source1_id, [x for x in ids.split(",") if x]))
            if len(batch) >= size:
                yield batch, None
                batch = []
    if batch:
        yield batch, None


def read_parquet_batches(directory, size):
    for path in sorted(glob.glob(os.path.join(directory, "*.parquet"))):
        frame = generic_metadata(pl.read_parquet(path)).sort(["source1_id", "target_id"])
        entities = frame["source1_id"].unique(maintain_order=True).to_list()
        for start in range(0, len(entities), size):
            chunk = frame.filter(pl.col("source1_id").is_in(entities[start:start + size]))
            lists = chunk.group_by("source1_id", maintain_order=True).agg(pl.col("target_id"))
            yield list(lists.iter_rows()), chunk


def load_truth(database, ground_truth):
    """Store train labels in the record store so batches can fetch their own."""
    connection = sqlite3.connect(database)
    if connection.execute("SELECT name FROM sqlite_master WHERE name = 'truth'").fetchone() is None:
        connection.execute("CREATE TABLE truth (source1_id TEXT PRIMARY KEY, matches TEXT)")
        with open(ground_truth, encoding="utf-8") as handle:
            handle.readline()
            connection.executemany("INSERT INTO truth VALUES (?, ?)",
                                   (line.rstrip("\n").split("\t") for line in handle))
        connection.commit()
    connection.close()


def fetch_truth(store, source1_ids):
    cursor = store.connection.cursor()
    cursor.execute("DROP TABLE IF EXISTS temp.labels")
    cursor.execute("CREATE TEMP TABLE labels (id TEXT PRIMARY KEY)")
    cursor.executemany("INSERT OR IGNORE INTO temp.labels VALUES (?)", ((i,) for i in source1_ids))
    cursor.execute("SELECT t.source1_id, t.matches FROM temp.labels l JOIN truth t ON t.source1_id = l.id")
    return {s: set(m.split(",")) - {""} for s, m in cursor.fetchall()}


WORKER = {}


def init_worker(database, index_dir, labelled, w, out_dir):
    WORKER.update(store=RecordStore(database), idf=IdfTable(index_dir), index_dir=index_dir,
                  labelled=labelled, w=w, out_dir=out_dir, idf_cache={})


def compute_batch(task):
    number, batch, metadata = task
    store = WORKER["store"]
    records = store.fetch({s for s, _ in batch} | {t for _, ts in batch for t in ts})
    pairs, offsets = [], [0]
    for source1_id, targets in batch:
        if source1_id in records:
            pairs.extend((source1_id, t) for t in targets if t in records)
        offsets.append(len(pairs))
    if not pairs:
        return 0
    columns = batch_features(pairs, records, WORKER["idf"], WORKER["index_dir"], WORKER["w"], WORKER["idf_cache"])
    sources = np.array([t[:2] for _, t in pairs])
    columns.update(group_features(columns, offsets, sources))
    validation = {s: is_validation(s) for s, _ in batch}
    table = pl.DataFrame({"source1_id": [s for s, _ in pairs], "target_id": [t for _, t in pairs],
                          "is_validation": [validation[s] for s, _ in pairs]})
    if metadata is not None:
        present = [c for c in META_COLUMNS if c in metadata.columns]
        table = table.join(metadata.select("source1_id", "target_id", *present), on=["source1_id", "target_id"], how="left")
    table = table.with_columns([pl.Series(c, columns[c]) for c in ALL_COLUMNS if c in columns])
    table = table.with_columns([pl.lit(None, dtype=pl.Float32).alias(c) for c in ALL_COLUMNS if c not in table.columns])
    table = table.with_columns([pl.col(c).cast(pl.Float32) for c in ALL_COLUMNS])
    if WORKER["labelled"]:
        truth = fetch_truth(store, [s for s, _ in batch])
        table = table.with_columns(pl.Series("label", [t in truth.get(s, ()) for s, t in pairs], dtype=pl.Int8))
    table.select("source1_id", "target_id", "is_validation", *ALL_COLUMNS,
                 *(["label"] if WORKER["labelled"] else [])).write_parquet(
        os.path.join(WORKER["out_dir"], f"batch_{number:05d}.parquet"))
    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Compute pair features for a candidate set.")
    parser.add_argument("--candidates", required=True, help="Candidate TSV, or a merge_candidates Parquet directory")
    parser.add_argument("--database", required=True, help="Record store from record_store.py")
    parser.add_argument("--index-dir", required=True, help="Forward blocking index (IDF tables)")
    parser.add_argument("--out-dir", required=True, help="One Parquet file per batch is written here.")
    parser.add_argument("--ground-truth", help="Train labels; adds a `label` column.")
    parser.add_argument("--w", type=float, default=0.6, help="C1 weight (the configuration's w)")
    parser.add_argument("--batch-rows", type=int, default=5_000, help="Source-1 entities per batch")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", os.cpu_count() or 4)))
    args = parser.parse_args()

    started = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.ground_truth:
        load_truth(args.database, args.ground_truth)
    RecordStore.warm(args.database)
    batches = (read_parquet_batches(args.candidates, args.batch_rows) if os.path.isdir(args.candidates)
               else read_tsv_batches(args.candidates, args.batch_rows))
    tasks = ((number, batch, metadata) for number, (batch, metadata) in enumerate(batches))
    total = 0
    with CONTEXT.Pool(args.workers, initializer=init_worker,
                      initargs=(args.database, args.index_dir, bool(args.ground_truth), args.w, args.out_dir)) as pool:
        for number, count in enumerate(bounded_imap(pool, compute_batch, tasks, 2 * args.workers)):
            total += count
            if number % 20 == 0:
                log(f"batch {number}: {total} pairs, {time.time() - started:.0f}s")
    log(f"done: {total} pairs in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
