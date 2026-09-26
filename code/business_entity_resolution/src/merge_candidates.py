#!/usr/bin/env python3
"""Union forward, reverse and embedding retrieval into the final candidate set.

A configuration maps rank columns to depths, e.g.
  {"rank_name": 20, "rank_address": 20, "rank_combined_060": 30,
   "rev_rank_combined_060": 3, "emb_rank": 5}
A pair is a candidate when any configured column ranks it within its depth.
Pairs found only by reverse or embedding passes are capped per Source-1
entity at the forward design size (sum of forward depths), so no list exceeds
twice the forward design; the cap keeps the best-ranked of them.

The same `selection` is used by the blocking lab and by production, so what
the lab measures is exactly what the matcher later scores.

Production: inputs are repartitioned by a hash of source1_id, each partition
is merged in memory, and the outputs are candidates/part-*.parquet (with
censored per-pass ranks, found_by bitmask and each target's best/runner-up
reverse scores) and candidate_pairs.tsv with one row per Source-1 entity.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time

import polars as pl

BIG = 32767
IDS = ("source1_id", "target_id")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def is_forward(column):
    return column.startswith("rank_")


def forward_passes(config):
    """blocking_knn --passes spec for the forward columns of a configuration."""
    specs = []
    for column, k in config.items():
        if not is_forward(column) or k <= 0:
            continue
        name = column[len("rank_"):]
        specs.append(f"combined@{int(name.split('_')[1]) / 100:g}:{k}" if name.startswith("combined_") else f"{name}:{k}")
    return ",".join(specs)


def active(config):
    return {c: k for c, k in config.items() if isinstance(k, int) and k > 0 and "rank" in c}


def selection(frame, config):
    """Rows of `frame` kept by `config`, with found_by bitmask and n_passes."""
    config = active(config)
    columns = [c for c in config if c in frame.columns]
    forward = [c for c in columns if is_forward(c)]
    other = [c for c in columns if not is_forward(c)]
    hits = {c: pl.col(c).fill_null(BIG) <= config[c] for c in columns}
    in_forward = pl.any_horizontal([hits[c] for c in forward]) if forward else pl.lit(False)
    in_other = pl.any_horizontal([hits[c] for c in other]) if other else pl.lit(False)
    frame = frame.with_columns(in_forward.alias("_fwd"), in_other.alias("_oth")).filter(pl.col("_fwd") | pl.col("_oth"))
    cap = sum(config[c] for c in forward)
    extra = frame.filter(~pl.col("_fwd"))
    if extra.height:
        key = pl.min_horizontal([pl.col(c).fill_null(BIG) for c in other])
        extra = (extra.with_columns(key.alias("_key")).sort(["source1_id", "_key", "target_id"])
                 .with_columns(pl.int_range(pl.len()).over("source1_id").alias("_pos"))
                 .filter(pl.col("_pos") < cap).drop("_key", "_pos"))
    frame = pl.concat([frame.filter(pl.col("_fwd")), extra])
    bits = [pl.when(hits[c]).then(1 << i).otherwise(0) for i, c in enumerate(columns)]
    return frame.with_columns(pl.sum_horizontal(bits).cast(pl.Int32).alias("found_by") if bits else pl.lit(0).alias("found_by"),
                              pl.sum_horizontal([hits[c].cast(pl.Int8) for c in columns]).cast(pl.Int8).alias("n_passes")
                              if columns else pl.lit(0).alias("n_passes")).drop("_fwd", "_oth")


def censor(frame, config):
    """Ranks beyond a column's depth (or missing) become depth + 1; unused columns are dropped."""
    config = active(config)
    rank_columns = [c for c in frame.columns if "rank" in c and c != "n_passes"]
    frame = frame.drop([c for c in rank_columns if c not in config])
    return frame.with_columns([pl.when(pl.col(c).fill_null(BIG) <= k).then(pl.col(c)).otherwise(k + 1)
                               .cast(pl.Int16).alias(c) for c, k in config.items() if c in frame.columns])


def combine(frames):
    """Outer-combine pair frames that share (source1_id, target_id): min rank, max score."""
    frames = [f for f in frames if f is not None and f.height]
    if not frames:
        return None
    stacked = pl.concat(frames, how="diagonal_relaxed")
    aggregations = [(pl.col(c).min() if "rank" in c else pl.col(c).max()).alias(c)
                    for c in stacked.columns if c not in IDS]
    return stacked.group_by(list(IDS)).agg(aggregations)


def reverse_pairs(frame):
    """Reverse rows (target_id, source1_id, rev_*) -> pair rows without target-level stats."""
    return frame.select([c for c in frame.columns if not c.startswith(("rev_best_", "rev_second_"))])


def target_stats(frame):
    stats = [c for c in frame.columns if c.startswith(("rev_best_", "rev_second_"))]
    return frame.group_by("target_id").agg([pl.col(c).first() for c in stats])


# ---------------------------------------------------------------- production

def repartition(source_dir, out_dir, partitions, transform=None):
    for path in sorted(glob.glob(os.path.join(source_dir, "*.parquet"))):
        frame = pl.read_parquet(path)
        if transform is not None:
            frame = transform(frame)
        frame = frame.with_columns((pl.col("source1_id").hash(seed=17) % partitions).alias("_p"))
        for (p,), part in frame.group_by("_p"):
            os.makedirs(os.path.join(out_dir, str(p)), exist_ok=True)
            part.drop("_p").write_parquet(os.path.join(out_dir, str(p), os.path.basename(path)))


def read_partition(root, p):
    paths = glob.glob(os.path.join(root, str(p), "*.parquet"))
    return pl.concat([pl.read_parquet(x) for x in paths], how="diagonal_relaxed") if paths else None


def main():
    parser = argparse.ArgumentParser(description="Union retrieval passes into candidate_pairs.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("passes", help="Print the forward --passes spec for a configuration.")
    p.add_argument("--config", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--config", required=True, help="JSON with rank-column depths (from blocking_lab select).")
    m.add_argument("--forward", required=True)
    m.add_argument("--reverse")
    m.add_argument("--embedding")
    m.add_argument("--source1", required=True, help="Source-1 TSV: every entity gets a TSV row.")
    m.add_argument("--out-dir", required=True, help="Writes part-*.parquet with metadata here.")
    m.add_argument("--tsv", required=True, help="candidate_pairs.tsv")
    m.add_argument("--work-dir", required=True)
    m.add_argument("--partitions", type=int, default=16)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    if args.command == "passes":
        print(forward_passes(config))
        return

    started = time.time()
    scratch = os.path.join(args.work_dir, "merge_tmp")
    shutil.rmtree(scratch, ignore_errors=True)
    repartition(args.forward, os.path.join(scratch, "forward"), args.partitions)
    stats = None
    if args.reverse:
        repartition(args.reverse, os.path.join(scratch, "reverse"), args.partitions, transform=reverse_pairs)
        stats = pl.concat([target_stats(pl.read_parquet(x)) for x in sorted(glob.glob(os.path.join(args.reverse, "*.parquet")))])
        stats = stats.group_by("target_id").agg(pl.all().first())
    if args.embedding:
        repartition(args.embedding, os.path.join(scratch, "embedding"), args.partitions)
    log("repartitioned inputs")

    os.makedirs(args.out_dir, exist_ok=True)
    written, pairs = set(), 0
    with open(args.tsv, "w", encoding="utf-8") as tsv:
        tsv.write("source1_entity_id\tcandidate_entity_ids\n")
        for p in range(args.partitions):
            frame = combine([read_partition(os.path.join(scratch, name), p) for name in ("forward", "reverse", "embedding")
                             if os.path.isdir(os.path.join(scratch, name))])
            if frame is None:
                continue
            frame = censor(selection(frame, config), config)
            if stats is not None:
                frame = frame.join(stats, on="target_id", how="left")
            frame.write_parquet(os.path.join(args.out_dir, f"part-{p:03d}.parquet"))
            for source1_id, targets in frame.group_by("source1_id").agg(pl.col("target_id").sort()).iter_rows():
                tsv.write(f"{source1_id}\t{','.join(targets)}\n")
                written.add(source1_id)
            pairs += frame.height
            log(f"partition {p}: {frame.height} pairs")
        entities = 0
        with open(args.source1, encoding="utf-8") as source:
            source.readline()
            for line in source:
                entities += 1
                source1_id = line.split("\t", 1)[0]
                if source1_id not in written:
                    tsv.write(f"{source1_id}\t\n")
    shutil.rmtree(scratch, ignore_errors=True)
    report = {"config": config, "entities": entities, "candidate_pairs": pairs,
              "mean_candidates": round(pairs / max(entities, 1), 2), "seconds": round(time.time() - started, 1)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
