"""Stdlib smoke test for the candidate-generation contract."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "candidate_generation.py"))
HEADER = "entity_id\tbusiness_name\tbusiness_address\tcountry\n"


def write_source(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(HEADER)
        handle.writelines("\t".join(row) + "\n" for row in rows)


class CandidateGenerationTest(unittest.TestCase):
    def test_open_country_union_and_train_recall(self):
        with tempfile.TemporaryDirectory() as root:
            train = os.path.join(root, "train")
            os.mkdir(train)
            write_source(os.path.join(train, "train_source1.tsv"), [
                ("S1-1", "Acme Corp", "10 Main Street, Paris, 75001", "France"),
                ("S1-2", "No Match", "Elsewhere", "France"),
            ])
            write_source(os.path.join(train, "train_source2.tsv"), [
                ("S2-1", "Acme Corporation", "10 Main Street, Paris, 75001", "FRANCE"),
                ("S2-2", "Acorn Cafe", "Other", "France"),
            ])
            write_source(os.path.join(train, "train_source3.tsv"), [
                ("S3-1", "Acme Corp", "Wrong country", "US"),
                ("S3-2", "Different", "Paris, 75001", "France"),
            ])
            with open(os.path.join(train, "train_ground_truth.tsv"), "w", encoding="utf-8", newline="") as handle:
                handle.write("source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1,S3-2\nS1-2\t\n")
            candidate = os.path.join(root, "candidate_pairs.tsv")
            report = os.path.join(root, "report.json")
            subprocess.run([sys.executable, SCRIPT, "--data-dir", root, "--split", "train",
                            "--output", candidate, "--report", report,
                            "--evaluate-ground-truth"], check=True, capture_output=True, text=True)
            with open(candidate, encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle, delimiter="\t"))
            self.assertEqual(rows[0], ["source1_entity_id", "candidate_entity_ids"])
            self.assertEqual(rows[1], ["S1-1", "S2-1,S3-2"])
            self.assertEqual(rows[2], ["S1-2", ""])
            with open(report, encoding="utf-8") as handle:
                result = json.load(handle)
            self.assertEqual(result["ground_truth_evaluation"]["candidate_recall_pct"], 100.0)


if __name__ == "__main__":
    unittest.main()
