"""Checks the scorer against the challenge's worked example and singleton rules."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from score import entity_f05, score  # noqa: E402
from split import is_validation  # noqa: E402


class ScoreTest(unittest.TestCase):
    def test_challenge_worked_example(self):
        predicted = {"S2-00047", "S2-00193", "S3-00812"}
        truth = {"S2-00047", "S3-00812"}
        self.assertAlmostEqual(entity_f05(predicted, truth), 0.714, places=3)

    def test_singleton_and_empty_rules(self):
        self.assertEqual(entity_f05(set(), set()), 1.0)
        self.assertEqual(entity_f05({"S2-1"}, set()), 0.0)
        self.assertEqual(entity_f05(set(), {"S2-1"}), 0.0)
        self.assertEqual(entity_f05({"S2-9"}, {"S2-1"}), 0.0)

    def test_missing_rows_are_empty_and_extra_rows_ignored(self):
        truth = {"S1-1": {"S2-1"}, "S1-2": set()}
        result = score({"S1-1": {"S2-1"}, "S1-99": {"S3-5"}}, truth)
        self.assertEqual(result["macro_f05"], 1.0)
        self.assertEqual(result["missing_prediction_rows"], 1)

    def test_split_is_deterministic(self):
        ids = [f"S1-{n}" for n in range(20000)]
        first = [is_validation(x) for x in ids]
        self.assertEqual(first, [is_validation(x) for x in ids])
        self.assertAlmostEqual(sum(first) / len(ids), 0.10, delta=0.01)


if __name__ == "__main__":
    unittest.main()
