"""Forward and reverse blocking on a tiny dataset: alias rows, C1 fusion, output contract."""

import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import polars as pl

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, SRC)

from blocking_knn import parse_passes, pass_matrix  # noqa: E402

HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
SOURCES = {
    1: [("S1-1", "Interstate Prudential PLLC", "117 Mistletoe Dr, Austin, TX", "US"),
        ("S1-2", "Lakshmi Consultancy Pvt Ltd", "12 MG Road, Pune", "India"),
        ("S1-3", "Verte Freres SARL", "82 Avenue du Marechal, La Baule", "France"),
        ("S1-4", "Cedar Market", "5 Oak St, Dallas, TX", "US")],
    2: [("S2-1", "Interstate Prudential P.L.L.C.", "117 Mistletoe Drive, Austin TX", "US"),
        ("S2-2", "लक्ष्मी कंसल्टेंसी प्राइवेट लिमिटेड", "12 M G Rd Pune", "India"),
        ("S2-3", "Cedar Mkt", "5 Oak Street Dallas", "US"),
        ("S2-4", "Random Other Inc", "999 Pine Ave, Boston", "US")],
    3: [("S3-1", "Pyrapyra aka Interstate Prudential PLLC", "", "US"),
        ("S3-2", "SARL VERTE & FRERES", "Loire-Atlantique, La Baule, 82 AVENUE DU MARECHAL", "France"),
        ("S3-3", "Zeta Corp", "1 Elm St", "US")],
}


def write_dataset(root):
    os.makedirs(os.path.join(root, "train"))
    for n, rows in SOURCES.items():
        with open(os.path.join(root, "train", f"train_source{n}.tsv"), "w", encoding="utf-8") as handle:
            handle.write(HEADER)
            handle.writelines("\t".join(row) + "\n" for row in rows)


def run(root, *args):
    subprocess.run([sys.executable, os.path.join(SRC, "blocking_knn.py"), "--data-dir", root, "--split", "train",
                    "--chunk-rows", "2", "--batch-rows", "3", "--workers", "2", *args],
                   check=True, capture_output=True, cwd=SRC)


class BlockingTest(unittest.TestCase):
    def test_combined_pass_equals_concatenated_vectors(self):
        rng = np.random.RandomState(0)
        qn, qa, tn, ta = (rng.rand(1, 6) for _ in range(4))
        qn, qa, tn, ta = (v / np.linalg.norm(v) for v in (qn, qa, tn, ta))
        w = 0.6
        concatenated = np.hstack([np.sqrt(w) * qn, np.sqrt(1 - w) * qa]) @ np.hstack([np.sqrt(w) * tn, np.sqrt(1 - w) * ta]).T
        import scipy.sparse as sp
        products = {"name": sp.csr_matrix(qn @ tn.T), "address": sp.csr_matrix(qa @ ta.T)}
        (_, weights, _), = parse_passes("combined@0.6:5")
        self.assertAlmostEqual(pass_matrix(products, weights).toarray()[0, 0], concatenated[0, 0], places=6)

    def test_forward_and_reverse_find_true_links_through_aliases_and_scripts(self):
        with tempfile.TemporaryDirectory() as root:
            write_dataset(root)
            run(root, "--index-dir", os.path.join(root, "idx"), "--pairs-out", os.path.join(root, "fwd"),
                "--output", os.path.join(root, "cand.tsv"), "--passes", "name:3,address:3,combined@0.6:3")
            forward = pl.read_parquet(os.path.join(root, "fwd", "*.parquet"))
            pairs = set(zip(forward["source1_id"], forward["target_id"]))
            for link in (("S1-1", "S2-1"), ("S1-1", "S3-1"), ("S1-2", "S2-2"), ("S1-3", "S3-2"), ("S1-4", "S2-3")):
                self.assertIn(link, pairs)
            self.assertEqual(forward.select("source1_id", "target_id").n_unique(), forward.height)
            row = forward.filter((pl.col("source1_id") == "S1-1") & (pl.col("target_id") == "S2-1")).row(0, named=True)
            self.assertAlmostEqual(row["score_combined_060"], 0.6 * row["score_name"] + 0.4 * row["score_address"], places=5)
            with open(os.path.join(root, "cand.tsv"), encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            self.assertEqual(len(lines), 5)

            run(root, "--direction", "reverse", "--index-dir", os.path.join(root, "idx_s1"),
                "--pairs-out", os.path.join(root, "rev"), "--passes", "name:2,address:2,combined@0.6:2")
            reverse = pl.read_parquet(os.path.join(root, "rev", "*.parquet"))
            alias = reverse.filter((pl.col("target_id") == "S3-1") & (pl.col("source1_id") == "S1-1")).row(0, named=True)
            self.assertEqual(alias["rev_rank_name"], 1)
            self.assertGreaterEqual(alias["rev_best_name"], alias["rev_second_name"] or 0.0)


if __name__ == "__main__":
    unittest.main()
