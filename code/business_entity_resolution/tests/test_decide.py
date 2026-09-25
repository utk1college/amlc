"""Decision rules: one owner per target, and a lower bar for an entity's best pair."""

import os
import sys
import unittest

import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from decide import decide, to_links  # noqa: E402


SCORES = pl.DataFrame({
    "source1_id": ["S1-a", "S1-a", "S1-b", "S1-b", "S1-c"],
    "target_id": ["S2-1", "S3-2", "S2-1", "S3-3", "S3-4"],
    "score": [0.9, 0.6, 0.7, 0.4, 0.2],
})


class DecideTest(unittest.TestCase):
    def test_one_owner_gives_shared_target_to_best_entity(self):
        links = to_links(decide(SCORES, threshold=0.5, top_threshold=0.5))
        self.assertEqual(links, {"S1-a": {"S2-1", "S3-2"}})

    def test_without_one_owner_both_entities_keep_shared_target(self):
        links = to_links(decide(SCORES, threshold=0.5, top_threshold=0.5, one_owner=False))
        self.assertEqual(links["S1-b"], {"S2-1"})

    def test_top_threshold_keeps_only_the_best_pair(self):
        links = to_links(decide(SCORES, threshold=0.5, top_threshold=0.3))
        self.assertEqual(links["S1-b"], {"S3-3"})
        self.assertNotIn("S1-c", links)


if __name__ == "__main__":
    unittest.main()
