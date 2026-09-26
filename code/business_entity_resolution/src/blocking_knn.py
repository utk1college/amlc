#!/usr/bin/env python3
"""TF-IDF nearest-neighbour candidate generation, forward and reverse, streamed in chunks.

Nothing is loaded whole. The indexed side is read line by line into
per-country chunks of --chunk-rows records; each chunk is featurized and
written to an on-disk index, then freed. The query side is read the same way
in batches of --batch-rows, and each batch is scored against the index one
chunk at a time, keeping only a running top-k per query and pass.

Views (each an L2-normalized TF-IDF vector with per-country IDF):
  name          character 3-grams of the normalized name and of its consonant
                skeleton, plus skeleton words and word pairs (N1)
  address       skeleton words and word pairs of the address (A1)
  address_char  character 3-grams of the normalized address (A2)

A pass ranks by one view, or by combined@w = w*name + (1-w)*address (C1).
Because each view is L2-normalized, that score is exactly the inner product
of the concatenated vectors [sqrt(w)*name | sqrt(1-w)*address], computed from
the same two sparse products the single-view passes already need.

Directions:
  forward   Source-1 queries against a Source-2/3 index. The index holds one
            row per name variant ("X aka Y" -> X aka Y, X, Y), all carrying
            the record's ID.
  reverse   Source-2/3 queries (one per name variant) against a Source-1
            index. Also records each target's best and runner-up score.

Output (--pairs-out, a directory of Parquet parts): one row per
(source1_id, target_id) with rank_<pass> / score_<pass> for every pass that
found the pair within its k (reverse columns are prefixed rev_).

Search cost is bounded per query: its features are used rarest first until
their document frequencies add up to --posting-budget.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import get_all_start_methods, get_context

import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from split import lab_set
from text_normalize import name_variants, normalize_address, normalize_name, skeleton

FEATURE_VERSION = 3
N_FEATURES = 2 ** 22
VIEWS = ("name", "address", "address_char")  # order of the stacked df/idf arrays
CONTEXT = get_context("fork" if "fork" in get_all_start_methods() else "spawn")


def char_grams(prefix, text):
    grams = []
    for word in text.split():
        padded = f" {word} "
        grams.extend(prefix + padded[i:i + 3] for i in range(len(padded) - 2))
    return grams


def name_features(normalized):
    skeleton_text = skeleton(normalized)
    features = char_grams("p", normalized) + char_grams("s", skeleton_text)
    words = skeleton_text.split()
    features.extend("w" + word for word in words)
    features.extend(f"b{a} {b}" for a, b in zip(words, words[1:]))
    return features


def address_features(normalized):
    words = skeleton(normalized).split()
    return words + [f"{a} {b}" for a, b in zip(words, words[1:])]


def address_char_features(normalized):
    return char_grams("c", normalized)


VECTORIZERS = {
    view: HashingVectorizer(analyzer=analyzer, n_features=N_FEATURES, alternate_sign=False,
                            norm=None, dtype=np.float32)
    for view, analyzer in (("name", name_features), ("address", address_features),
                           ("address_char", address_char_features))
}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def parse_passes(spec):
    """"name:20,address:20,address_char:10,combined@0.6:30" -> [(name, view weights, k)]."""
    passes = []
    for item in spec.split(","):
        label, k = item.strip().rsplit(":", 1)
        if label.startswith("combined@"):
            w = float(label.split("@", 1)[1])
            passes.append((f"combined_{round(w * 100):03d}", {"name": w, "address": 1 - w}, int(k)))
        elif label in VIEWS:
            passes.append((label, {label: 1.0}, int(k)))
        else:
            raise ValueError(f"unknown pass {label!r}")
    return passes


def prepare(rows, variants):
    """(id, raw name, raw address) rows -> ids, names, addresses; one row per name variant."""
    ids, names, addresses = [], [], []
    for entity_id, name, address in rows:
        normalized_address = normalize_address(address)
        for variant in (name_variants(name) if variants else [normalize_name(name)]):
            ids.append(entity_id)
            names.append(variant)
            addresses.append(normalized_address)
    return ids, names, addresses


def featurize(names, addresses, views):
    texts = {"name": names, "address": addresses, "address_char": addresses}
    return {view: VECTORIZERS[view].transform(texts[view]) for view in views}


def stream_batches(paths, size, keep_ids=None):
    """Yield (country, rows) with at most `size` rows of one country per batch."""
    buffers = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                entity_id, name, address, country = line.rstrip("\n").split("\t")
                if keep_ids is not None and not keep_ids(entity_id):
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
    """Featurize one chunk, store raw counts, return its row count and document frequencies."""
    prefix, rows, views, variants = task
    ids, names, addresses = prepare(rows, variants)
    counts = featurize(names, addresses, views)
    for view, matrix in counts.items():
        sp.save_npz(f"{prefix}.{view}.npz", matrix, compressed=False)
    np.save(prefix + ".ids.npy", np.array(ids, dtype="S"))
    return len(ids), np.stack([np.bincount(counts[v].indices, minlength=N_FEATURES).astype(np.int32)
                               for v in views])


def finalize_chunk(task):
    """Weight a chunk by its country's IDF and store it transposed for search."""
    prefix, idf_path, views = task
    idf = np.load(idf_path)
    for i, view in enumerate(views):
        counts = sp.load_npz(f"{prefix}.{view}.npz")
        sp.save_npz(f"{prefix}.{view}.T.npz", tfidf(counts, idf[i]).T.tocsr(), compressed=False)
        os.remove(f"{prefix}.{view}.npz")


def build_index(paths, index_dir, views, variants, args, pool):
    manifest_path = os.path.join(index_dir, "manifest.json")
    wanted = {"feature_version": FEATURE_VERSION, "chunk_rows": args.chunk_rows,
              "views": list(views), "variants": variants}
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if all(manifest.get(k) == v for k, v in wanted.items()):
            log(f"reusing index {index_dir}")
            return manifest
    os.makedirs(index_dir, exist_ok=True)
    countries, pending = {}, []

    def collect(item):
        result, chunk, entry = item
        rows, df = result.get()
        chunk["rows"] = rows
        entry["df"] += df

    for country, rows in stream_batches(paths, args.chunk_rows):
        entry = countries.setdefault(country, {"chunks": [], "df": np.zeros((len(views), N_FEATURES), np.int64)})
        prefix = os.path.join(index_dir, f"{country.replace(' ', '_')}_{len(entry['chunks']):04d}")
        chunk = {"prefix": prefix, "rows": 0}
        entry["chunks"].append(chunk)
        pending.append((pool.apply_async(index_chunk, ((prefix, rows, views, variants),)), chunk, entry))
        while len(pending) >= args.workers + 2:
            collect(pending.pop(0))
        log(f"queued {country} chunk {len(entry['chunks'])}")
    for item in pending:
        collect(item)

    tasks, manifest = [], {**wanted, "countries": {}}
    for country, entry in countries.items():
        stem = os.path.join(index_dir, country.replace(" ", "_"))
        offset = 0
        for chunk in entry["chunks"]:
            chunk["offset"] = offset
            offset += chunk["rows"]
        np.save(stem + ".df.npy", entry["df"])
        idf = np.log((offset + 1) / (entry["df"] + 1)) + 1
        np.save(stem + ".idf.npy", idf.astype(np.float32))
        tasks.extend((chunk["prefix"], stem + ".idf.npy", views) for chunk in entry["chunks"])
        manifest["countries"][country] = {"rows": offset, "stem": stem, "chunks": entry["chunks"]}
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


def pass_matrix(products, weights):
    """Weighted sum of per-view cosine products: the score of one pass."""
    matrix = None
    for view, weight in weights.items():
        term = products[view] if weight == 1 else products[view] * weight
        matrix = term if matrix is None else matrix + term
    return matrix.tocsr()


def empty_state(rows, passes):
    return {name: (np.full((rows, k), -1.0, np.float32), np.full((rows, k), -1, np.int64))
            for name, _, k in passes}


def search_chunk(task):
    """Score a slice of queries against one index chunk; return per-pass top-k arrays."""
    q_views, row_offset, prefix, offset, passes, sub_batch = task
    needed = sorted({view for _, weights, _ in passes for view in weights})
    index = {view: sp.load_npz(f"{prefix}.{view}.T.npz") for view in needed}
    rows = q_views[needed[0]].shape[0]
    out = empty_state(rows, passes)
    for start in range(0, rows, sub_batch):
        end = min(start + sub_batch, rows)
        products = {view: (q_views[view][start:end] @ index[view]).tocsr() for view in needed}
        for name, weights, _ in passes:
            fill_top_k(pass_matrix(products, weights), *out[name], start, offset)
    return row_offset, out


def merge(state, update):
    scores = np.concatenate([state[0], update[0]], axis=1)
    index = np.concatenate([state[1], update[1]], axis=1)
    keep = np.argpartition(-scores, state[0].shape[1] - 1, axis=1)[:, :state[0].shape[1]]
    return np.take_along_axis(scores, keep, 1), np.take_along_axis(index, keep, 1)


def search_batch(q_views, rows, country_index, passes, args, pool):
    """Per-pass (scores, index row) arrays, best first. Tasks = query slices x index chunks."""
    chunks = country_index["chunks"]
    slices = [s for s in np.array_split(np.arange(rows), max(1, -(-args.workers // len(chunks)))) if len(s)]
    state = empty_state(rows, passes)
    tasks = []
    for piece in slices:
        start, end = int(piece[0]), int(piece[-1]) + 1
        sliced = {view: matrix[start:end] for view, matrix in q_views.items()}
        tasks.extend((sliced, start, chunk["prefix"], chunk["offset"], passes, args.sub_batch) for chunk in chunks)
    for row_offset, update in pool.imap_unordered(search_chunk, tasks):
        for name, _, _ in passes:
            n = update[name][0].shape[0]
            window = slice(row_offset, row_offset + n)
            merged = merge((state[name][0][window], state[name][1][window]), update[name])
            state[name][0][window], state[name][1][window] = merged
    for name in state:
        order = np.argsort(-state[name][0], axis=1, kind="stable")
        state[name] = tuple(np.take_along_axis(a, order, 1) for a in state[name])
    return state


def to_pairs(state, query_ids, index_ids, passes, target_stats):
    """Wide frame: one row per (query_id, index_id) with rank_<pass>, score_<pass>.

    Rows of the same record (name variants) are collapsed to their best score,
    then ranks are recomputed per query and pass and cut at that pass's k.
    """
    parts = []
    for name, _, _ in passes:
        scores, index = state[name]
        rows, cols = np.nonzero(index >= 0)
        if len(rows):
            parts.append(pl.DataFrame({"query_id": query_ids.gather(rows),
                                       "index_id": index_ids.gather(index[rows, cols]),
                                       "score": scores[rows, cols]}).with_columns(pl.lit(name).alias("pass")))
    columns = [c for name, _, _ in passes for c in (f"rank_{name}", f"score_{name}")]
    if target_stats:
        columns += [c for name, _, _ in passes for c in (f"best_{name}", f"second_{name}")]
    if not parts:
        return pl.DataFrame(schema={"query_id": pl.Utf8, "index_id": pl.Utf8,
                                    **{c: (pl.Int16 if c.startswith("rank_") else pl.Float32) for c in columns}})
    long = (pl.concat(parts).group_by(["query_id", "pass", "index_id"]).agg(pl.col("score").max())
            .with_columns(pl.col("score").rank("ordinal", descending=True).over(["query_id", "pass"]).alias("rank"))
            .join(pl.DataFrame({"pass": [n for n, _, _ in passes], "k": [k for _, _, k in passes]}), on="pass")
            .filter(pl.col("rank") <= pl.col("k")))
    aggregations = []
    for name, _, _ in passes:
        is_pass = pl.col("pass") == name
        aggregations += [pl.col("rank").filter(is_pass).min().cast(pl.Int16).alias(f"rank_{name}"),
                         pl.col("score").filter(is_pass).max().cast(pl.Float32).alias(f"score_{name}")]
    wide = long.group_by(["query_id", "index_id"]).agg(aggregations)
    if target_stats:
        ordered = pl.col("score").sort(descending=True)
        stats = long.group_by(["query_id", "pass"]).agg(ordered.first().alias("best"),
                                                        ordered.slice(1, 1).first().alias("second"))
        for name, _, _ in passes:
            wide = wide.join(stats.filter(pl.col("pass") == name).select(
                "query_id", pl.col("best").cast(pl.Float32).alias(f"best_{name}"),
                pl.col("second").cast(pl.Float32).alias(f"second_{name}")), on="query_id", how="left")
    return wide.select("query_id", "index_id", *columns)


def peak_rss_gb():
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        return round(usage / 1024 ** 2, 2)
    except (ImportError, AttributeError):
        return None


def read_id_filter(args):
    if args.lab_only:
        return lambda entity_id: lab_set(entity_id) is not None
    if args.source1_ids:
        with open(args.source1_ids, encoding="utf-8") as handle:
            handle.readline()
            wanted = {line.split("\t", 1)[0].rstrip("\n") for line in handle}
        return wanted.__contains__
    return None


def main():
    parser = argparse.ArgumentParser(description="Chunked TF-IDF nearest-neighbour blocking.")
    parser.add_argument("--data-dir", required=True, help="Dataset directory containing train/ and test/.")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--direction", choices=("forward", "reverse"), default="forward")
    parser.add_argument("--index-dir", required=True, help="On-disk index; reused when compatible.")
    parser.add_argument("--passes", default="name:20,address:20,combined@0.6:30",
                        help="Comma list of view:k or combined@w:k.")
    parser.add_argument("--pairs-out", help="Directory of Parquet parts with per-pass ranks and scores.")
    parser.add_argument("--output", help="Forward only: candidate TSV (union of passes, every Source-1 row).")
    parser.add_argument("--report", help="Destination JSON report.")
    parser.add_argument("--lab-only", action="store_true", help="Forward: query only lab-set Source-1 entities.")
    parser.add_argument("--source1-ids", help="Forward: ground-truth-style TSV; only its Source-1 IDs are queried.")
    parser.add_argument("--posting-budget", type=int, default=50_000,
                        help="Max summed document frequency of the features one query searches with.")
    parser.add_argument("--chunk-rows", type=int, default=250_000)
    parser.add_argument("--batch-rows", type=int, default=20_000)
    parser.add_argument("--sub-batch", type=int, default=200)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", os.cpu_count() or 4)))
    args = parser.parse_args()

    started = time.time()
    passes = parse_passes(args.passes)
    split_dir = os.path.join(args.data_dir, args.split)
    source1 = [os.path.join(split_dir, f"{args.split}_source1.tsv")]
    targets = [os.path.join(split_dir, f"{args.split}_source{n}.tsv") for n in (2, 3)]
    forward = args.direction == "forward"
    index_paths, index_variants, index_views = (targets, True, VIEWS) if forward else (source1, False, ("name", "address"))
    query_paths, query_variants = (source1, False) if forward else (targets, True)
    query_views = sorted({view for _, weights, _ in passes for view in weights})
    missing = set(query_views) - set(index_views)
    if missing:
        raise ValueError(f"passes need views {sorted(missing)} that the {args.direction} index does not build")

    with CONTEXT.Pool(args.workers, maxtasksperchild=50) as pool:
        manifest = build_index(index_paths, args.index_dir, index_views, index_variants, args, pool)
    index_seconds = round(time.time() - started, 1)

    keep = read_id_filter(args) if forward else None
    if args.pairs_out:
        os.makedirs(args.pairs_out, exist_ok=True)
    tsv = open(args.output, "w", encoding="utf-8") if (args.output and forward) else None
    if tsv:
        tsv.write("source1_entity_id\tcandidate_entity_ids\n")
    ids_cache, per_country, pairs_written, queries, part = {}, {}, 0, 0, 0
    with CONTEXT.Pool(args.workers, maxtasksperchild=200) as pool:
        for country, rows in stream_batches(query_paths, args.batch_rows, keep):
            batch_started = time.time()
            queries += len(rows)
            country_index = manifest["countries"].get(country)
            if country_index is None:
                if tsv:
                    tsv.writelines(f"{row[0]}\t\n" for row in rows)
                continue
            if country not in ids_cache:
                ids_cache.clear()
                ids_cache[country] = pl.Series(np.concatenate(
                    [np.load(chunk["prefix"] + ".ids.npy") for chunk in country_index["chunks"]]).astype(str))
            df = np.load(country_index["stem"] + ".df.npy")
            idf = np.load(country_index["stem"] + ".idf.npy")
            ids, names, addresses = prepare(rows, query_variants)
            counts = featurize(names, addresses, query_views)
            q_views = {view: budgeted(tfidf(counts[view], idf[index_views.index(view)]),
                                      df[index_views.index(view)], args.posting_budget) for view in query_views}
            state = search_batch(q_views, len(ids), country_index, passes, args, pool)
            frame = to_pairs(state, pl.Series(ids), ids_cache[country], passes, target_stats=not forward)
            if forward:
                frame = frame.rename({"query_id": "source1_id", "index_id": "target_id"})
            else:
                frame = frame.rename({"query_id": "target_id", "index_id": "source1_id",
                                      **{c: "rev_" + c for c in frame.columns if c not in ("query_id", "index_id")}})
            if args.pairs_out and frame.height:
                frame.write_parquet(os.path.join(args.pairs_out, f"part-{part:05d}.parquet"))
                part += 1
            if tsv:
                lists = dict(frame.group_by("source1_id").agg(pl.col("target_id").sort()).iter_rows())
                tsv.writelines(f"{row[0]}\t{','.join(lists.get(row[0], []))}\n" for row in rows)
            pairs_written += frame.height
            stats = per_country.setdefault(country, {"queries": 0, "seconds": 0.0})
            stats["queries"] += len(rows)
            stats["seconds"] = round(stats["seconds"] + time.time() - batch_started, 1)
            log(f"{args.direction} {country}: {stats['queries']} queries ({time.time() - batch_started:.1f}s batch)")
    if tsv:
        tsv.close()

    report = {"parameters": vars(args), "passes": [(n, w, k) for n, w, k in passes],
              "index_seconds": index_seconds, "countries": per_country, "queries": queries,
              "pair_rows": pairs_written, "rows_per_query": round(pairs_written / max(queries, 1), 2),
              "seconds": round(time.time() - started, 1), "peak_rss_gb": peak_rss_gb()}
    print(json.dumps(report, indent=2))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
