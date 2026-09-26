"""Lab metrics and the shared union rule, on hand-computed cases."""

import os
import sys
import unittest

import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from blocking_lab import LabSet  # noqa: E402
from merge_candidates import censor, forward_passes, selection  # noqa: E402

ENTITIES = pl.DataFrame({"source1_id": ["A", "B", "C"], "lab_set": ["block_dev"] * 3, "country": ["us"] * 3})
TRUTH = pl.DataFrame({"source1_id": ["A", "A", "C"], "target_id": ["S2-1", "S3-2", "S2-3"]})
PAIRS = pl.DataFrame({
    "source1_id": ["A", "A", "A", "C"],
    "target_id": ["S2-1", "S3-2", "S2-9", "S2-3"],
    "rank_name": [1, 30, 2, None],
    "rank_address": [None, None, None, 5],
    "is_true": [True, True, False, True],
}, schema_overrides={"rank_name": pl.Int16, "rank_address": pl.Int16})
SLICES = TRUTH.with_columns(pl.lit("us").alias("country"), pl.col("target_id").str.slice(0, 2).alias("source"),
                            pl.lit("latin").alias("script"), pl.lit(False).alias("empty_address"),
                            pl.lit(False).alias("alias"))


class LabMetricsTest(unittest.TestCase):
    def setUp(self):
        self.lab = LabSet("block_dev", ENTITIES, TRUTH, PAIRS, SLICES)

    def test_ceiling_recall_and_candidates_match_hand_computation(self):
        narrow = self.lab.summary(self.lab.evaluate({"rank_name": 20, "rank_address": 0}))
        # A: 1 of 2 links -> 1.25*0.5/(0.25+0.5); B singleton -> 1; C: 0.
        self.assertAlmostEqual(narrow["ceiling_f05"], (1.25 * 0.5 / 0.75 + 1 + 0) / 3, places=6)
        self.assertAlmostEqual(narrow["link_recall"], 1 / 3, places=6)
        self.assertAlmostEqual(narrow["mean_candidates"], 2 / 3, places=2)
        wide = self.lab.summary(self.lab.evaluate({"rank_name": 50, "rank_address": 10}))
        self.assertEqual(wide["ceiling_f05"], 1.0)

    def test_paired_gain(self):
        a = self.lab.evaluate({"rank_name": 20})
        b = self.lab.evaluate({"rank_name": 50, "rank_address": 10})
        gain = self.lab.paired(a, b)
        self.assertAlmostEqual(gain["ceiling_gain"], 1 - (1.25 * 0.5 / 0.75 + 1) / 3, places=6)
        self.assertAlmostEqual(gain["recall_gain"], 2 / 3, places=6)
        self.assertGreater(gain["ceiling_se"], 0)

    def test_v2_pairs_are_scored_like_configurations(self):
        v2 = pl.DataFrame({"source1_id": ["A", "C"], "target_id": ["S2-1", "S2-3"]})
        self.assertAlmostEqual(self.lab.summary(self.lab.evaluate_pairs(v2))["link_recall"], 2 / 3, places=6)


class UnionRuleTest(unittest.TestCase):
    def test_reverse_only_additions_are_capped_at_forward_depth(self):
        frame = pl.DataFrame({
            "source1_id": ["E"] * 5,
            "target_id": ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5"],
            "rank_name": [1, None, None, None, None],
            "rev_rank_name": [None, 1, 1, 2, 1],
        }, schema_overrides={"rank_name": pl.Int16, "rev_rank_name": pl.Int16})
        chosen = selection(frame, {"rank_name": 2, "rev_rank_name": 3})
        # Forward depth sum is 2, so at most 2 reverse-only pairs join the forward hit.
        self.assertEqual(chosen.height, 3)
        self.assertEqual(set(chosen["target_id"]), {"S2-1", "S2-2", "S2-3"})
        self.assertEqual(chosen.filter(pl.col("target_id") == "S2-1")["found_by"].item(), 1)
        censored = censor(chosen, {"rank_name": 2, "rev_rank_name": 3})
        self.assertEqual(censored.filter(pl.col("target_id") == "S2-2")["rank_name"].item(), 3)

    def test_forward_passes_spec(self):
        spec = forward_passes({"rank_name": 20, "rank_address": 0, "rank_combined_060": 30, "rev_rank_name": 3})
        self.assertEqual(spec, "name:20,combined@0.6:30")


if __name__ == "__main__":
    unittest.main()
