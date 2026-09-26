"""Smoke tests for the leave-one-country-out and adversarial checks on synthetic feature files."""

import argparse
import os
import sys
import tempfile
import unittest

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from pair_features import FEATURE_COLUMNS  # noqa: E402
from robustness import adversarial, loco  # noqa: E402
from split import is_validation  # noqa: E402


def write_split(directory, prefix, countries, entities_per_country, shift):
    """Write one feature Parquet and a Source-1 TSV; returns the TSV path."""
    rng = np.random.RandomState(3)
    os.makedirs(directory, exist_ok=True)
    rows, source1 = [], ["entity_id\tbusiness_name\tbusiness_address\tcountry"]
    for country in countries:
        for n in range(entities_per_country):
            source1_id = f"S1-{prefix}{country}{n}"
            source1.append(f"{source1_id}\tname\taddr\t{country}")
            for k in range(2):
                features = rng.rand(len(FEATURE_COLUMNS)).astype(np.float32) * 100
                features[FEATURE_COLUMNS.index("blend")] += shift
                rows.append({"source1_id": source1_id, "target_id": f"S2-{prefix}{country}{n}-{k}",
                             "is_validation": is_validation(source1_id), "label": int(k == 0),
                             **dict(zip(FEATURE_COLUMNS, features))})
    pl.DataFrame(rows).write_parquet(os.path.join(directory, "batch_00000.parquet"))
    path = os.path.join(directory, "source1.tsv")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(source1) + "\n")
    return path


class RobustnessTest(unittest.TestCase):
    def test_adversarial_flags_a_shifted_country_but_not_a_matching_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            val_tsv = write_split(os.path.join(tmp, "val"), "v", ["same"], 5000, 0)
            # Same country label on both sides; only the second test set is drifted.
            test_tsv = write_split(os.path.join(tmp, "test"), "t", ["same"], 5000, 0)
            args = argparse.Namespace(val_features_dir=os.path.join(tmp, "val"),
                                      test_features_dir=os.path.join(tmp, "test"),
                                      val_source1=val_tsv, test_source1=test_tsv, per_country=400, threads=2)
            report = adversarial(args)
            self.assertLess(report["same"]["auc"], 0.65)

            shifted_tsv = write_split(os.path.join(tmp, "test2"), "u", ["same"], 5000, 60)
            args.test_features_dir, args.test_source1 = os.path.join(tmp, "test2"), shifted_tsv
            drift = adversarial(args)["same"]
            self.assertGreater(drift["auc"], 0.9)
            self.assertEqual(drift["top_drifting_features"][0][0], "blend")

    def test_loco_runs_and_reports_the_metric(self):
        with tempfile.TemporaryDirectory() as tmp:
            tsv = write_split(os.path.join(tmp, "f"), "x", ["us", "india"], 3000, 0)
            truth = os.path.join(tmp, "truth.tsv")
            with open(truth, "w", encoding="utf-8") as handle:
                handle.write("source1_entity_id\tmatched_entity_ids\n")
                for country in ("us", "india"):
                    for n in range(3000):
                        handle.write(f"S1-x{country}{n}\tS2-x{country}{n}-0\n")
            args = argparse.Namespace(features_dir=os.path.join(tmp, "f"), source1=tsv, truth=truth,
                                      train_country="us", eval_country="india", train_fraction=1.0,
                                      rounds=3, threshold=0.5, top_threshold=None, threads=2)
            result = loco(args)
            self.assertEqual(result["eval_country"], "india")
            self.assertGreater(result["entities"], 0)
            self.assertIn("macro_f05", result)


if __name__ == "__main__":
    unittest.main()
