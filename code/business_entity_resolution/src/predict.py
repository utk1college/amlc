#!/usr/bin/env python3
"""Score feature batches with a trained matcher, one batch at a time.

Only pairs scoring at least --floor are written. No decision rule in
decide.py can select a pair below its thresholds, so any floor at or below
the lowest threshold in use gives identical matches from a much smaller file.
"""

import argparse
import glob
import os
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pair_features import FEATURE_COLUMNS
from train_matcher import matrix


def main():
    parser = argparse.ArgumentParser(description="Score candidate-pair features.")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scores-out", required=True)
    parser.add_argument("--floor", type=float, default=0.3)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    model = lgb.Booster(model_file=args.model)
    writer, scored, kept, started = None, 0, 0, time.time()
    for path in sorted(glob.glob(os.path.join(args.features_dir, "*.parquet"))):
        table = pq.read_table(path, columns=["source1_id", "target_id", *FEATURE_COLUMNS])
        scores = model.predict(matrix(table), num_threads=args.threads).astype(np.float32)
        keep = pa.array(scores >= args.floor)
        out = pa.table({"source1_id": table.column("source1_id"), "target_id": table.column("target_id"),
                        "score": scores}).filter(keep)
        writer = writer or pq.ParquetWriter(args.scores_out, out.schema)
        writer.write_table(out)
        scored += len(scores)
        kept += out.num_rows
    if writer:
        writer.close()
    print(f"scored {scored} pairs, kept {kept} at score >= {args.floor}, {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
