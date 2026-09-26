#!/usr/bin/env python3
"""Score feature batches with the mean of the fold models, one batch at a time.

Test scores are the average of the same fold models whose out-of-fold scores
Stage 2 and the decision rule were tuned on, so both share one distribution.
Only pairs scoring at least --floor are written; any floor at or below the
lowest threshold later used gives identical decisions from a smaller file.
"""

import argparse
import glob
import os
import time

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from train_matcher import feature_list


def main():
    parser = argparse.ArgumentParser(description="Score candidate-pair features with the fold models.")
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--model-dir", required=True, help="Directory with fold*.txt from train_matcher.py")
    parser.add_argument("--features", choices=("v3", "v2"), default="v3")
    parser.add_argument("--scores-out", required=True)
    parser.add_argument("--floor", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("WORKERS", os.cpu_count() or 4)))
    args = parser.parse_args()

    columns = feature_list(args.features)
    models = [lgb.Booster(model_file=path) for path in sorted(glob.glob(os.path.join(args.model_dir, "fold*.txt")))]
    if not models:
        raise SystemExit(f"no fold models in {args.model_dir}")
    writer, scored, kept, started = None, 0, 0, time.time()
    for path in sorted(glob.glob(os.path.join(args.features_dir, "*.parquet"))):
        table = pq.read_table(path, columns=["source1_id", "target_id", *columns])
        matrix = np.column_stack([table.column(c).to_numpy(zero_copy_only=False) for c in columns]).astype(np.float32)
        scores = np.mean([m.predict(matrix, num_threads=args.threads) for m in models], axis=0).astype(np.float32)
        keep = pa.array(scores >= args.floor)
        out = pa.table({"source1_id": table.column("source1_id"), "target_id": table.column("target_id"),
                        "score": scores}).filter(keep)
        writer = writer or pq.ParquetWriter(args.scores_out, out.schema)
        writer.write_table(out)
        scored += len(scores)
        kept += out.num_rows
    if writer:
        writer.close()
    print(f"scored {scored} pairs with {len(models)} fold models, kept {kept} at score >= {args.floor}, "
          f"{time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
