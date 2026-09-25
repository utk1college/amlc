#!/usr/bin/env python3
"""Deterministic entity-level train/validation split of the training labels.

The split is a pure function of the Source-1 ID, so every stage (blocking,
feature building, threshold tuning) agrees on it without sharing a file.
"""

import argparse
import hashlib
import os
import sys


VALIDATION_FRACTION = 0.10
HEADER = "source1_entity_id\tmatched_entity_ids\n"


def is_validation(source1_id, fraction=VALIDATION_FRACTION):
    digest = hashlib.md5(source1_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64 < fraction


def main():
    parser = argparse.ArgumentParser(description="Split train ground truth by Source-1 entity.")
    parser.add_argument("--ground-truth", required=True, help="train_ground_truth.tsv")
    parser.add_argument("--out-dir", required=True, help="Writes train_ground_truth.tsv and val_ground_truth.tsv here.")
    parser.add_argument("--fraction", type=float, default=VALIDATION_FRACTION)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    counts = {"train": [0, 0], "val": [0, 0]}  # [entities, singletons]
    with open(args.ground_truth, encoding="utf-8") as source, \
            open(os.path.join(args.out_dir, "train_ground_truth.tsv"), "w", encoding="utf-8") as train, \
            open(os.path.join(args.out_dir, "val_ground_truth.tsv"), "w", encoding="utf-8") as val:
        if source.readline() != HEADER:
            raise ValueError(f"{args.ground_truth}: invalid ground-truth header")
        train.write(HEADER)
        val.write(HEADER)
        for line in source:
            source1_id, _, matches = line.rstrip("\n").partition("\t")
            part = "val" if is_validation(source1_id, args.fraction) else "train"
            (val if part == "val" else train).write(line)
            counts[part][0] += 1
            counts[part][1] += not matches
    for part, (entities, singletons) in counts.items():
        print(f"{part}: {entities} entities, {singletons} singletons ({100 * singletons / entities:.2f}%)")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
