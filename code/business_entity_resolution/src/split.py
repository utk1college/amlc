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
# Disjoint lab sets inside the non-validation entities, as shares of a
# separately salted hash: about 50k / 50k / 100k of the 1.99M entities.
LAB_SETS = (("block_dev", 0.025), ("block_confirm", 0.05), ("e2e_confirm", 0.10))
OOF_FOLDS = 3


def unit_hash(key):
    digest = hashlib.md5(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def is_validation(source1_id, fraction=VALIDATION_FRACTION):
    return unit_hash(source1_id) < fraction


def lab_set(source1_id):
    """block_dev, block_confirm, e2e_confirm or None; never a validation entity."""
    if is_validation(source1_id):
        return None
    u = unit_hash("lab:" + source1_id)
    for name, upper in LAB_SETS:
        if u < upper:
            return name
    return None


def oof_fold(source1_id, folds=OOF_FOLDS, salt="fold:"):
    """Entity-grouped fold of a non-validation entity; None for validation."""
    if is_validation(source1_id):
        return None
    return min(int(unit_hash(salt + source1_id) * folds), folds - 1)


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
