"""Diagnostics classify links as scored, rejected or never-candidate, and compare val with test."""

import os
import sys
import unittest

import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from diagnostics import loss_report, shift_report  # noqa: E402

SCORES = pl.DataFrame({
    "source1_id": ["S1-a", "S1-a", "S1-a", "S1-b", "S1-c"],
    "target_id": ["S2-1", "S2-2", "S3-9", "S3-3", "S2-4"],
    "score": [0.95, 0.60, 0.90, 0.85, 0.40],
})
TRUTH = {"S1-a": {"S2-1", "S2-2", "S3-8"}, "S1-b": {"S3-3"}, "S1-c": set()}
COUNTRIES = {"S1-a": "us", "S1-b": "india", "S1-c": "india"}
NAMES = {"S2-1": "acme", "S2-2": "acme", "S3-8": "अक्मे", "S3-9": "other", "S3-3": "beta", "S2-4": "gamma"}


class DiagnosticsTest(unittest.TestCase):
    def test_loss_report_separates_rejected_missing_and_false_links(self):
        report = loss_report(SCORES, TRUTH, COUNTRIES, NAMES, threshold=0.75, top_threshold=0.75)
        slices = report["links_by_country_source_nonascii"]
        self.assertEqual(slices["us/S2/False"], {"tp": 1, "rejected": 1, "not_candidate": 0})
        self.assertEqual(slices["us/S3/True"]["not_candidate"], 1)
        self.assertEqual(slices["us/S3/False"]["false_positive"], 1)
        self.assertEqual(report["by_country"]["india"]["singletons"], 1)
        self.assertEqual(report["by_country"]["india"]["singleton_loss"], 0.0)

    def test_shift_report_counts_empty_lists_per_country(self):
        test_scores = pl.DataFrame({"source1_id": ["T-1", "T-1"], "target_id": ["S2-1", "S3-1"], "score": [0.9, 0.8]})
        report = shift_report(SCORES, test_scores, COUNTRIES, {"T-1": "us", "T-2": "us"}, 0.3, 0.75, 0.75)
        self.assertEqual(report["india"]["val"]["empty_list_share"], 0.5)
        self.assertEqual(report["us"]["test_all"]["links_per_entity"], 1.0)
        self.assertEqual(report["us"]["test_all"]["empty_list_share"], 0.5)


if __name__ == "__main__":
    unittest.main()
