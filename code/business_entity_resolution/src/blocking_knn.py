#!/usr/bin/env python3
"""TF-IDF nearest-neighbour candidate generation, streamed in chunks.

Nothing is loaded whole. Source 2/3 are read line by line into per-country
chunks of --chunk-rows records; each chunk is featurized and written to an
on-disk index, then freed. Source 1 is read the same way in batches of
--batch-rows queries, and each batch is scored against the index one target
chunk at a time, keeping only a running top-k per query.

Within a country, three rankings are kept: by name, by address, and by a
weighted sum of both. A name is represented by character 3-grams of its
normalized text and of its consonant skeleton plus skeleton words and word
pairs, so typos, transliterations and reorderings all land close together.

Search cost is bounded per query: its features are used rarest first until
their document frequencies add up to --posting-budget, so one query can
touch at most that many target records whatever its name looks like.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from text_normalize import normalize_address, normalize_name, skeleton

FEATURE_VERSION = 2
N_FEATURES = 2 ** 22
PASSES = ("name", "address", "combined")
RECALL_CUTOFFS = (1, 3, 5, 10, 20, 50, 100)


def name_features(normalized):
    skeleton_text = skeleton(normalized)
    features = []
    for prefix, text in (("p", normalized), ("s", skeleton_text)):
        for word in text.split():
            padded = f" {word} "
            features.extend(prefix + padded[i:i + 3] for i in range(len(padded) - 2))
    words = skeleton_text.split()
    features.extend("w" + word for word in words)
    features.extend(f"b{a} {b}" for a, b in zip(words, words[1:]))
    return features


def address_features(normalized):
    words = skeleton(normalized).split()
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


NAME = HashingVectorizer(analyzer=name_features, n_features=N_FEATURES,
                         alternate_sign=False, norm=None, dtype=np.float32)
ADDRESS = HashingVectorizer(analyzer=address_features, n_features=N_FEATURES,
                            alternate_sign=False, norm=None, dtype=np.float32)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def featurize(rows):
    """Raw feature counts for (entity_id, name, address) rows."""
    return (NAME.transform([normalize_name(name) for _, name, _ in rows]),
            ADDRESS.transform([normalize_address(address) for _, _, address in rows]))


def stream_batches(paths, size, keep_ids=None):
    """Yield (country, rows) with at most `size` rows of one country per batch."""
    buffers = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                entity_id, name, address, country = line.rstrip("\n").split("\t")
                if keep_ids is not None and entity_id not in keep_ids:
                    continue
                country = country.strip().casefold()
                buffer = buffers.setdefault(country, [])
                buffer.append((entity_id, name, address))
                if len(buffer) >= size:
                    yield country, buffers.pop(country)
    yield from buffers.items()


def tfidf(counts, idf):
    counts.data *= idf[counts.indices]
    return l2_normalize(counts, copy=False)


# ---------------------------------------------------------------- index build

def index_chunk(task):
    """Featurize one target chunk, store raw counts, return its document frequencies."""
    prefix, rows = task
    names, addresses = featurize(rows)
    sp.save_npz(prefix + ".name.npz", names, compressed=False)
    sp.save_npz(prefix + ".address.npz", addresses, compressed=False)
    np.save(prefix + ".ids.npy", np.array([row[0] for row in rows], dtype="S"))
    return (np.bincount(names.indices, minlength=N_FEATURES).astype(np.int32),
            np.bincount(addresses.indices, minlength=N_FEATURES).astype(np.int32))


def finalize_chunk(task):
    """Weight a chunk by its country's IDF and store it transposed for search."""
    prefix, idf_path = task
    idf = np.load(idf_path)
    for view, weights in (("name", idf[0]), ("address", idf[1])):
        counts = sp.load_npz(f"{prefix}.{view}.npz")
        sp.save_npz(f"{prefix}.{view}.T.npz", tfidf(counts, weights).T.tocsr(), compressed=False)
        os.remove(f"{prefix}.{view}.npz")


def build_index(target_paths, index_dir, args, pool):
    manifest_path = os.path.join(index_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("feature_version") == FEATURE_VERSION and manifest.get("chunk_rows") == args.chunk_rows:
            log(f"reusing index {index_dir}")
            return manifest
    os.makedirs(index_dir, exist_ok=True)
    countries, pending = {}, []

    def collect(result, country):
        name_df, address_df = result
        entry = countries[country]
        entry["name_df"] += name_df
        entry["address_df"] += address_df

    for country, rows in stream_batches(target_paths, args.chunk_rows):
        entry = countries.setdefault(country, {"chunks": [], "rows": 0,
                                               "name_df": np.zeros(N_FEATURES, np.int64),
                                               "address_df": np.zeros(N_FEATURES, np.int64)})
        prefix = os.path.join(index_dir, f"{country.replace(' ', '_')}_{len(entry['chunks']):04d}")
        entry["chunks"].append({"prefix": prefix, "offset": entry["rows"], "rows": len(rows)})
        entry["rows"] += len(rows)
        pending.append((pool.apply_async(index_chunk, ((prefix, rows),)), country))
        while len(pending) >= args.workers + 2:
            result, owner = pending.pop(0)
            collect(result.get(), owner)
        log(f"queued {country} chunk {len(entry['chunks'])} ({entry['rows']} rows)")
    for result, owner in pending:
        collect(result.get(), owner)

    tasks, manifest = [], {"feature_version": FEATURE_VERSION, "chunk_rows": args.chunk_rows, "countries": {}}
    for country, entry in countries.items():
        stem = os.path.join(index_dir, country.replace(" ", "_"))
        n = entry["rows"]
        np.save(stem + ".df.npy", np.stack([entry["name_df"], entry["address_df"]]))
        idf = np.log((n + 1) / (np.stack([entry["name_df"], entry["address_df"]]) + 1)) + 1
        np.save(stem + ".idf.npy", idf.astype(np.float32))
        tasks.extend((chunk["prefix"], stem + ".idf.npy") for chunk in entry["chunks"])
        manifest["countries"][country] = {"rows": n, "stem": stem, "chunks": entry["chunks"]}
    pool.map(finalize_chunk, tasks)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1)
    log(f"index built: { {c: v['rows'] for c, v in manifest['countries'].items()} }")
    return manifest


# ---------------------------------------------------------------- search

def budgeted(queries, df, budget):
    """Keep each query's rarest features until their frequencies reach the budget."""
    feature_df = df[queries.indices]
    keep = np.zeros(len(feature_df), dtype=bool)
    for i in range(queries.shape[0]):
        start, end = queries.indptr[i], queries.indptr[i + 1]
        if start == end:
            continue
        order = np.argsort(feature_df[start:end], kind="stable")
        within = np.cumsum(feature_df[start:end][order]) <= budget
        within[0] = True
        keep[start + order[within]] = True
    queries.data[~keep] = 0
    queries.eliminate_zeros()
    return queries


def fill_top_k(matrix, scores_out, index_out, row_offset, column_offset):
    k = scores_out.shape[1]
    for i in range(matrix.shape[0]):
        start, end = matrix.indptr[i], matrix.indptr[i + 1]
        if start == end:
            continue
        scores, columns = matrix.data[start:end], matrix.indices[start:end]
        if end - start > k:
            keep = np.argpartition(-scores, k)[:k]
            scores, columns = scores[keep], columns[keep]
        scores_out[row_offset + i, :len(scores)] = scores
        index_out[row_offset + i, :len(scores)] = columns + column_offset


def search_chunk(task):
    """Score a query batch against one target chunk; return per-pass top-k arrays."""
    q_name, q_address, prefix, offset, k, name_weight, sub_batch = task
    t_name = sp.load_npz(prefix + ".name.T.npz")
    t_address = sp.load_npz(prefix + ".address.T.npz")
    rows = q_name.shape[0]
    out = {p: (np.full((rows, k[p]), -1.0, np.float32), np.full((rows, k[p]), -1, np.int64)) for p in PASSES}
    for start in range(0, rows, sub_batch):
        end = min(start + sub_batch, rows)
        by_name = (q_name[start:end] @ t_name).tocsr()
        by_address = (q_address[start:end] @ t_address).tocsr()
        combined = (by_name * name_weight + by_address * (1 - name_weight)).tocsr()
        for p, matrix in zip(PASSES, (by_name, by_address, combined)):
            fill_top_k(matrix, *out[p], start, offset)
    return out


def merge(state, update):
    scores = np.concatenate([state[0], update[0]], axis=1)
    index = np.concatenate([state[1], update[1]], axis=1)
    keep = np.argpartition(-scores, state[0].shape[1] - 1, axis=1)[:, :state[0].shape[1]]
    return np.take_along_axis(scores, keep, 1), np.take_along_axis(index, keep, 1)


def search_batch(rows, country_index, args, pool):
    """Return per-pass (scores, target_index) arrays sorted best first."""
    df = np.load(country_index["stem"] + ".df.npy")
    idf = np.load(country_index["stem"] + ".idf.npy")
    names, addresses = featurize(rows)
    q_name = budgeted(tfidf(names, idf[0]), df[0], args.posting_budget)
    q_address = budgeted(tfidf(addresses, idf[1]), df[1], args.posting_budget)
    k = {"name": args.k_name, "address": args.k_address, "combined": args.k_combined}
    state = {p: (np.full((len(rows), k[p]), -1.0, np.float32), np.full((len(rows), k[p]), -1, np.int64))
             for p in PASSES}
    tasks = [(q_name, q_address, chunk["prefix"], chunk["offset"], k, args.name_weight, args.sub_batch)
             for chunk in country_index["chunks"]]
    for update in pool.imap_unordered(search_chunk, tasks):
        for p in PASSES:
            state[p] = merge(state[p], update[p])
    for p in PASSES:
        order = np.argsort(-state[p][0], axis=1, kind="stable")
        state[p] = (np.take_along_axis(state[p][0], order, 1), np.take_along_axis(state[p][1], order, 1))
    return state


def read_truth(path):
    truth = {}
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            source1_id, _, matches = line.rstrip("\n").partition("\t")
            truth[source1_id] = {x for x in matches.split(",") if x}
    return truth


def main():
    parser = argparse.ArgumentParser(description="Chunked TF-IDF nearest-neighbour blocking.")
    parser.add_argument("--data-dir", required=True, help="Dataset directory containing train/ and test/.")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output", required=True, help="Destination candidate_pairs TSV.")
    parser.add_argument("--index-dir", required=True, help="On-disk target index; reused when compatible.")
    parser.add_argument("--report", help="Destination JSON report.")
    parser.add_argument("--source1-ids", help="Ground-truth-style TSV; only its Source-1 IDs are queried.")
    parser.add_argument("--truth", help="Ground truth for a per-pass recall@k report.")
    parser.add_argument("--k-name", type=int, default=20)
    parser.add_argument("--k-address", type=int, default=20)
    parser.add_argument("--k-combined", type=int, default=30)
    parser.add_argument("--name-weight", type=float, default=0.6)
    parser.add_argument("--posting-budget", type=int, default=50_000,
                        help="Max summed document frequency of the features one query searches with.")
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--batch-rows", type=int, default=20_000)
    parser.add_argument("--sub-batch", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    started = time.time()
    split_dir = os.path.join(args.data_dir, args.split)
    context = get_context("fork")
    with context.Pool(args.workers, maxtasksperchild=50) as pool:
        manifest = build_index([os.path.join(split_dir, f"{args.split}_source{n}.tsv") for n in (2, 3)],
                               args.index_dir, args, pool)
    index_seconds = round(time.time() - started, 1)

    keep_ids = set(read_truth(args.source1_ids)) if args.source1_ids else None
    truth = read_truth(args.truth) if args.truth else None
    hits = {p: dict.fromkeys(RECALL_CUTOFFS, 0) for p in PASSES}
    ids_cache, per_country, pairs, rows_written = {}, {}, 0, 0
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle, \
            context.Pool(args.workers, maxtasksperchild=200) as pool:
        handle.write("source1_entity_id\tcandidate_entity_ids\n")
        source1 = [os.path.join(split_dir, f"{args.split}_source1.tsv")]
        for country, rows in stream_batches(source1, args.batch_rows, keep_ids):
            batch_started = time.time()
            country_index = manifest["countries"].get(country)
            if country_index is None:
                handle.writelines(f"{row[0]}\t\n" for row in rows)
                rows_written += len(rows)
                continue
            if country not in ids_cache:
                ids_cache.clear()
                ids_cache[country] = np.concatenate([np.load(chunk["prefix"] + ".ids.npy")
                                                     for chunk in country_index["chunks"]])
            target_ids = ids_cache[country]
            state = search_batch(rows, country_index, args, pool)
            for i, row in enumerate(rows):
                ranked = {p: [j for j in state[p][1][i] if j >= 0] for p in PASSES}
                union = sorted({target_ids[j].decode() for p in PASSES for j in ranked[p]})
                handle.write(f"{row[0]}\t{','.join(union)}\n")
                pairs += len(union)
                if truth is not None:
                    true_ids = truth.get(row[0], set())
                    for p in PASSES:
                        found = [target_ids[j].decode() for j in ranked[p]]
                        for cutoff in RECALL_CUTOFFS:
                            hits[p][cutoff] += len(true_ids.intersection(found[:cutoff]))
            rows_written += len(rows)
            stats = per_country.setdefault(country, {"queries": 0, "seconds": 0.0})
            stats["queries"] += len(rows)
            stats["seconds"] = round(stats["seconds"] + time.time() - batch_started, 1)
            log(f"{country}: {stats['queries']} queries done ({time.time() - batch_started:.1f}s this batch)")

    report = {"parameters": vars(args), "index_seconds": index_seconds, "countries": per_country,
              "source1_rows": rows_written, "candidate_pairs": pairs,
              "mean_candidates": round(pairs / max(rows_written, 1), 2),
              "seconds": round(time.time() - started, 1)}
    if truth is not None:
        links = sum(len(v) for v in truth.values())
        report["recall_at_k_by_pass"] = {p: {f"@{c}": round(100 * v / links, 2) for c, v in row.items()}
                                         for p, row in hits.items()}
    print(json.dumps(report, indent=2))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
