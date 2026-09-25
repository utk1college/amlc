#!/usr/bin/env python3
"""Pairwise features for candidate pairs, computed batch by batch.

The candidate file is read --batch-rows Source-1 entities at a time; only the
records of that batch are fetched from the record store, their pair features
are computed (string similarity with RapidFuzz, IDF-weighted token overlap,
house-number agreement, and within-entity ranks), and one Parquet file is
written per batch. Nothing whole is held in memory.
"""

import argparse
import math
import os
import sqlite3
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
from sklearn.utils import murmurhash3_32

from record_store import RecordStore
from split import is_validation

N_FEATURES = 2 ** 22  # must match blocking_knn.N_FEATURES
LEGAL_WORDS = {"private", "limited", "incorporated", "corporation", "company", "llc", "llp",
               "lp", "plc", "pllc", "pc", "sarl", "sas", "sa", "eurl", "gmbh", "the", "and"}
WORKERS = 6


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


class IdfTable:
    """IDF of name skeleton words and address words, per country, from the blocking index."""

    def __init__(self, index_dir):
        self.index_dir, self.tables, self.cache = index_dir, {}, {}

    def weight(self, country, view, token):
        key = (country, view, token)
        value = self.cache.get(key)
        if value is None:
            if country not in self.tables:
                path = os.path.join(self.index_dir, country.replace(" ", "_") + ".idf.npy")
                self.tables[country] = np.load(path) if os.path.exists(path) else None
            table = self.tables[country]
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


def python_features(a, b, idf):
    """Per-pair features that need token logic (records are store rows)."""
    country = a[1]
    a_words, b_words = a[3].split(), b[3].split()
    a_core = [w for w in a[2].split() if w not in LEGAL_WORDS]
    b_core = [w for w in b[2].split() if w not in LEGAL_WORDS]
    a_legal = {w for w in a[2].split() if w in LEGAL_WORDS - {"the", "and"}}
    b_legal = {w for w in b[2].split() if w in LEGAL_WORDS - {"the", "and"}}
    name_jaccard, name_shared, name_missing, name_extra = idf_overlap(
        a_words, b_words, lambda t: idf.weight(country, 0, t))
    a_address, b_address = a[5].split(), b[5].split()
    address_jaccard, address_shared, address_missing, address_extra = idf_overlap(
        a_address, b_address, lambda t: idf.weight(country, 1, t))
    a_numbers, b_numbers = numbers(a[4].split()), numbers(b[4].split())
    number_union = set(a_numbers) | set(b_numbers)
    first_gap = (math.log1p(abs(int(a_numbers[0][:9]) - int(b_numbers[0][:9])))
                 if a_numbers and b_numbers else -1.0)
    return (
        name_jaccard, name_shared, name_missing, name_extra,
        float(" ".join(a_core) == " ".join(b_core)),
        float(bool(a_core) and bool(b_core) and a_core[0] == b_core[0]),
        len(a_words), len(b_words),
        float(bool(a_legal)), float(bool(b_legal)),
        float(bool(a_legal) and bool(b_legal) and not (a_legal & b_legal)),
        address_jaccard, address_shared, address_missing, address_extra,
        float(not a[4]), float(not b[4]),
        len(set(a_numbers) & set(b_numbers)) / len(number_union) if number_union else -1.0,
        float(bool(a_numbers) and bool(b_numbers) and a_numbers[0] == b_numbers[0]),
        first_gap,
        float(b[6]), float(b[0].startswith("S3")),
    )


PYTHON_COLUMNS = (
    "name_idf_jaccard", "name_idf_shared", "name_idf_missing_max", "name_idf_extra_max",
    "core_equal", "first_core_equal", "name_words_s1", "name_words_target",
    "legal_s1", "legal_target", "legal_conflict",
    "address_idf_jaccard", "address_idf_shared", "address_idf_missing_max", "address_idf_extra_max",
    "address_missing_s1", "address_missing_target",
    "number_jaccard", "first_number_equal", "first_number_log_gap",
    "target_non_ascii", "target_source3",
)
FUZZY_COLUMNS = {
    # column: (scorer, S1 field, target field) — fields index the store row
    "name_ratio": (fuzz.ratio, 2), "name_partial": (fuzz.partial_ratio, 2),
    "name_token_sort": (fuzz.token_sort_ratio, 2), "name_token_set": (fuzz.token_set_ratio, 2),
    "name_jaro_winkler": (JaroWinkler.normalized_similarity, 2),
    "skeleton_ratio": (fuzz.ratio, 3), "skeleton_token_set": (fuzz.token_set_ratio, 3),
    "address_ratio": (fuzz.ratio, 4), "address_partial": (fuzz.partial_ratio, 4),
    "address_token_set": (fuzz.token_set_ratio, 4), "address_skeleton_token_set": (fuzz.token_set_ratio, 5),
}
GROUP_SOURCES = ("name_token_set", "address_token_set", "blend")
FEATURE_COLUMNS = (tuple(FUZZY_COLUMNS) + PYTHON_COLUMNS + ("blend",)
                   + tuple(f"{c}_rank" for c in GROUP_SOURCES)
                   + tuple(f"{c}_gap_to_best" for c in GROUP_SOURCES)
                   + ("group_size", "group_strong_names"))


def group_features(columns, offsets):
    """Rank and gap-to-best of each pair within its Source-1 entity's candidates."""
    n = len(columns["blend"])
    out = {f"{c}_rank": np.zeros(n, np.float32) for c in GROUP_SOURCES}
    out.update({f"{c}_gap_to_best": np.zeros(n, np.float32) for c in GROUP_SOURCES})
    out["group_size"] = np.zeros(n, np.float32)
    out["group_strong_names"] = np.zeros(n, np.float32)
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
    return out


def batch_features(pairs, records, idf):
    """pairs: list of (source1_id, target_id), grouped by source1_id."""
    a_rows = [records[s] for s, _ in pairs]
    b_rows = [records[t] for _, t in pairs]
    columns = {}
    for name, (scorer, field) in FUZZY_COLUMNS.items():
        columns[name] = cpdist([r[field] for r in a_rows], [r[field] for r in b_rows],
                               scorer=scorer, workers=WORKERS).astype(np.float32)
    columns["name_jaro_winkler"] *= 100
    python = np.array([python_features(a, b, idf) for a, b in zip(a_rows, b_rows)], dtype=np.float32)
    for i, name in enumerate(PYTHON_COLUMNS):
        columns[name] = python[:, i]
    columns["blend"] = (0.6 * columns["name_token_set"] + 0.4 * columns["address_token_set"]).astype(np.float32)
    return columns


def read_candidate_batches(path, size, keep):
    batch = []
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            source1_id, _, ids = line.rstrip("\n").partition("\t")
            if keep(source1_id):
                batch.append((source1_id, [x for x in ids.split(",") if x]))
                if len(batch) >= size:
                    yield batch
                    batch = []
    if batch:
        yield batch


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


def main():
    parser = argparse.ArgumentParser(description="Compute pair features for a candidate file.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--database", required=True, help="Record store from record_store.py")
    parser.add_argument("--index-dir", required=True, help="Blocking index (for IDF tables)")
    parser.add_argument("--out-dir", required=True, help="One Parquet file per batch is written here.")
    parser.add_argument("--ground-truth", help="Train labels; adds a `label` column.")
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="Share of non-validation Source-1 entities to keep (validation is always kept).")
    parser.add_argument("--batch-rows", type=int, default=20_000)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.ground_truth:
        load_truth(args.database, args.ground_truth)
    RecordStore.warm(args.database)
    store, idf = RecordStore(args.database), IdfTable(args.index_dir)

    def keep(source1_id):
        if args.train_fraction >= 1 or is_validation(source1_id):
            return True
        return is_validation("t" + source1_id, args.train_fraction)

    total = 0
    for number, batch in enumerate(read_candidate_batches(args.candidates, args.batch_rows, keep)):
        started = time.time()
        records = store.fetch({s for s, _ in batch} | {t for _, ts in batch for t in ts})
        pairs, offsets = [], [0]
        for source1_id, targets in batch:
            pairs.extend((source1_id, t) for t in targets if t in records)
            offsets.append(len(pairs))
        columns = batch_features(pairs, records, idf)
        columns.update(group_features(columns, offsets))
        table = {"source1_id": [s for s, _ in pairs], "target_id": [t for _, t in pairs],
                 "is_validation": [is_validation(s) for s, _ in pairs]}
        table.update({c: columns[c] for c in FEATURE_COLUMNS})
        if args.ground_truth:
            truth = fetch_truth(store, [s for s, _ in batch])
            table["label"] = np.array([t in truth.get(s, ()) for s, t in pairs], dtype=np.int8)
        pq.write_table(pa.table(table), os.path.join(args.out_dir, f"batch_{number:05d}.parquet"))
        total += len(pairs)
        log(f"batch {number}: {len(batch)} entities, {len(pairs)} pairs, {time.time() - started:.1f}s (total {total})")
    store.close()


if __name__ == "__main__":
    main()
