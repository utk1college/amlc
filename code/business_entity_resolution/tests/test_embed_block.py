"""Embedding search and efSearch tuning on synthetic unit vectors (no model download needed)."""

import argparse
import os
import sys
import tempfile
import unittest

import numpy as np
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    import faiss  # noqa: F401
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False

from split import lab_set  # noqa: E402


def unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


@unittest.skipUnless(HAVE_FAISS, "faiss not installed")
class EmbedBlockTest(unittest.TestCase):
    def test_search_finds_the_owner_and_tune_matches_exact(self):
        from embed_block import search, tune
        rng = np.random.default_rng(0)
        s1_ids = [f"S1-{n}" for n in range(4000)]
        dev = [i for i in s1_ids if lab_set(i) == "block_dev"][:60]
        s1 = unit(rng.normal(size=(len(s1_ids), 32)))
        owners = [s1_ids.index(i) for i in dev]
        targets = unit(s1[owners] + 0.05 * rng.normal(size=(len(owners), 32)))
        with tempfile.TemporaryDirectory() as root:
            cache = os.path.join(root, "emb")
            os.makedirs(cache)
            pl.DataFrame({"id": s1_ids, "country": ["india"] * len(s1_ids)}).write_parquet(os.path.join(cache, "source1.ids.parquet"))
            np.save(os.path.join(cache, "source1.name.npy"), s1)
            target_ids = [f"S2-{n}" for n in range(len(owners))]
            pl.DataFrame({"id": target_ids, "country": ["india"] * len(owners)}).write_parquet(os.path.join(cache, "targets.ids.parquet"))
            np.save(os.path.join(cache, "targets.name.npy"), targets)
            os.makedirs(os.path.join(root, "train"))
            with open(os.path.join(root, "train", "train_ground_truth.tsv"), "w", encoding="utf-8") as handle:
                handle.write("source1_entity_id\tmatched_entity_ids\n")
                handle.writelines(f"{s}\t{t}\n" for s, t in zip(dev, target_ids))

            self.assertTrue(np.allclose(np.linalg.norm(s1, axis=1), 1.0, atol=1e-5))
            common = dict(data_dir=root, out_dir=cache, form="name", threads=2, k=5)
            tuned = tune(argparse.Namespace(**common, ef_grid=[16, 64]))
            self.assertEqual(tuned["links"], len(dev))
            self.assertEqual(tuned["exact_recall"], 1.0)
            self.assertIsNotNone(tuned["chosen_ef_search"])
            search(argparse.Namespace(**common, ef_search=tuned["chosen_ef_search"], pairs_out=os.path.join(root, "pairs")))
            pairs = pl.read_parquet(os.path.join(root, "pairs", "*.parquet"))
            top = dict(pairs.filter(pl.col("emb_rank") == 1).select("target_id", "source1_id").iter_rows())
            self.assertEqual(top, dict(zip(target_ids, dev)))
            # Inner products of unit vectors are cosines: within [-1, 1], near 1 for the owner.
            self.assertLessEqual(float(pairs["emb_score"].max()), 1.0 + 1e-5)
            self.assertGreater(float(pairs.filter(pl.col("emb_rank") == 1)["emb_score"].min()), 0.9)


if __name__ == "__main__":
    unittest.main()
