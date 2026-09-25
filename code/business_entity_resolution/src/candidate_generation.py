#!/usr/bin/env python3
"""Generate audited, exact-union candidate lists for the ER challenge.

This is intentionally a blocking baseline, not a matching model.  It uses the
same four key definitions assessed by the initial EDA, avoids external data,
and writes the precise candidate list that a later model must score.
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone


LEGAL_SUFFIXES = {
    "llc", "inc", "incorporated", "corp", "corporation", "ltd", "limited",
    "pvt", "private", "co", "company", "llp", "pc", "pllc", "gmbh", "sa",
    "sarl", "sas", "enterprises", "enterprise", "services", "service",
    "solutions", "solution", "holdings", "holding", "group", "industries",
    "industry", "technologies", "technology", "consulting", "ventures",
}
RE_US_ZIP = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
RE_IN_PIN = re.compile(r"\b([1-9]\d{5})\b")
RE_FR_POST = re.compile(r"\b((?:0[1-9]|[1-8]\d|9[0-8])\d{3})\b")
RE_KEYWORD_POSTAL = re.compile(
    r"(?:\scode|pincode|pin|zip\scode|zipcode|zip|postal\scode|postcode)"
    r"\s[:#.-]?\s*(\d{4,6})", re.IGNORECASE
)


def normalize_text(text):
    """Use the EDA's reproducible normalization; retain legal terms for keys."""
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    text = re.sub(r"&", " and ", text.lower())
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def soundex(value):
    value = re.sub(r"[^a-zA-Z]", "", value).upper()
    if not value:
        return ""
    codes = {"B": "1", "F": "1", "P": "1", "V": "1", "C": "2", "G": "2",
             "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
             "D": "3", "T": "3", "L": "4", "M": "5", "N": "5", "R": "6"}
    output, previous = [value[0]], codes.get(value[0], "0")
    for character in value[1:]:
        code = codes.get(character, "0")
        if code != "0" and code != previous:
            output.append(code)
        previous = code
        if len(output) == 4:
            break
    return "".join(output).ljust(4, "0")


def extract_postal(address, country):
    """EDA-equivalent generic/open-set postal extractor."""
    if not address:
        return ""
    keyword = RE_KEYWORD_POSTAL.search(address)
    if keyword:
        return keyword.group(1)
    # Shapes are selected only to match the existing EDA. Unknown countries use
    # every shape; no country is rejected or specially filtered.
    country = country.upper()
    patterns = {"US": [RE_US_ZIP], "INDIA": [RE_IN_PIN], "FRANCE": [RE_FR_POST]}
    for pattern in patterns.get(country, [RE_IN_PIN, RE_US_ZIP, RE_FR_POST]):
        match = pattern.search(address.strip()[-15:])
        if match:
            return match.group(1)
    return ""


def blocking_keys(name, address, country):
    """Return non-empty (kind, normalized_country, key) triples."""
    normalized_country = country.strip().casefold()
    normalized_name = normalize_text(name)
    keys = []
    if normalized_name:
        words = normalized_name.split()
        if len(normalized_name) >= 3:
            keys.append(("first3", normalized_country, normalized_name[:3]))
        if words:
            keys.append(("firstword", normalized_country, words[0]))
        code = soundex(normalized_name)
        if code:
            keys.append(("soundex", normalized_country, code))
    postal = extract_postal(address, country)
    if postal:
        keys.append(("postal", normalized_country, postal))
    return keys


def records(path):
    """Yield validated source records from an uncompressed UTF-8 TSV."""
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        expected = ["entity_id", "business_name", "business_address", "country"]
        if header != expected:
            raise ValueError(f"{path}: expected header {expected}, got {header}")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 4:
                raise ValueError(f"{path}: row {line_number} has {len(row)} columns, expected 4")
            yield row


def build_index(connection, target_paths):
    connection.execute("CREATE TABLE blocks (kind TEXT, country TEXT, key TEXT, entity_id TEXT)")
    inserted, target_records, key_counts = 0, 0, Counter()
    batch = []
    for path in target_paths:
        for entity_id, name, address, country in records(path):
            target_records += 1
            for kind, normalized_country, key in blocking_keys(name, address, country):
                batch.append((kind, normalized_country, key, entity_id))
                key_counts[kind] += 1
            if len(batch) >= 50_000:
                connection.executemany("INSERT INTO blocks VALUES (?, ?, ?, ?)", batch)
                inserted += len(batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO blocks VALUES (?, ?, ?, ?)", batch)
            inserted += len(batch)
            batch.clear()
    connection.execute("CREATE INDEX blocks_lookup ON blocks(kind, country, key, entity_id)")
    connection.commit()
    return inserted, target_records, dict(key_counts)


def fetch_candidates(connection, keys):
    """Deduplicate candidates across the exact disjunction of all active keys."""
    candidates = set()
    for kind, country, key in keys:
        rows = connection.execute(
            "SELECT entity_id FROM blocks WHERE kind = ? AND country = ? AND key = ?",
            (kind, country, key),
        )
        candidates.update(row[0] for row in rows)
    return sorted(candidates)


def write_candidates(connection, source1_path, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    totals, active_keys = Counter(), Counter()
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for entity_id, name, address, country in records(source1_path):
            keys = blocking_keys(name, address, country)
            active_keys.update(kind for kind, _, _ in keys)
            candidates = fetch_candidates(connection, keys)
            writer.writerow([entity_id, ",".join(candidates)])
            totals["source1_rows"] += 1
            totals["candidate_pairs"] += len(candidates)
            if candidates:
                totals["source1_with_candidates"] += 1
            else:
                totals["source1_without_candidates"] += 1
    return dict(totals), dict(active_keys)


def evaluate_recall(candidate_path, ground_truth_path):
    """Measure exact candidate recall against all labelled train links."""
    candidates = {}
    with open(candidate_path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        next(reader, None)
        for source1_id, candidate_ids in reader:
            candidates[source1_id] = set(candidate_ids.split(",")) if candidate_ids else set()
    total, recovered = 0, 0
    with open(ground_truth_path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        expected = ["source1_entity_id", "matched_entity_ids"]
        if next(reader, None) != expected:
            raise ValueError(f"{ground_truth_path}: invalid ground-truth header")
        for source1_id, linked_ids in reader:
            truth = linked_ids.split(",") if linked_ids else []
            total += len(truth)
            recovered += sum(target in candidates.get(source1_id, set()) for target in truth)
    return {"labelled_pairs": total, "recovered_labelled_pairs": recovered,
            "candidate_recall_pct": round(100 * recovered / total, 6) if total else None}


def main():
    parser = argparse.ArgumentParser(description="Generate exact-union ER candidate pairs.")
    parser.add_argument("--data-dir", required=True, help="Dataset directory containing train/ and test/.")
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--output", required=True, help="Destination candidate_pairs TSV.")
    parser.add_argument("--report", required=True, help="Destination measured JSON report.")
    parser.add_argument("--work-dir", help="Directory for the temporary SQLite block index. Defaults to the output directory.")
    parser.add_argument("--evaluate-ground-truth", action="store_true",
                        help="Allowed only for --split train; measure candidate recall on all train labels.")
    args = parser.parse_args()
    if args.evaluate_ground_truth and args.split != "train":
        parser.error("--evaluate-ground-truth requires --split train")

    split_dir = os.path.join(args.data_dir, args.split)
    source1 = os.path.join(split_dir, f"{args.split}_source1.tsv")
    targets = [os.path.join(split_dir, f"{args.split}_source2.tsv"),
               os.path.join(split_dir, f"{args.split}_source3.tsv")]
    missing = [path for path in [source1, *targets] if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    work_dir = args.work_dir or os.path.dirname(os.path.abspath(args.output))
    os.makedirs(work_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="er_blocks_", dir=work_dir) as temporary_dir:
        database_path = os.path.join(temporary_dir, "blocks.sqlite")
        connection = sqlite3.connect(database_path)
        try:
            index_rows, target_records, indexed_by_kind = build_index(connection, targets)
            totals, source_keys = write_candidates(connection, source1, args.output)
        finally:
            connection.close()

    total_possible = totals["source1_rows"] * target_records
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "data_dir": os.path.abspath(args.data_dir),
        "blocking_keys": ["country+first3", "country+firstword", "country+soundex", "country+postal_if_present"],
        "target_index_rows": index_rows,
        "target_records": target_records,
        "target_index_rows_by_key": indexed_by_kind,
        "source1_active_keys": source_keys,
        **totals,
        "all_cross_source_comparisons": total_possible,
        "candidate_reduction_ratio_pct": round(100 * (1 - totals["candidate_pairs"] / total_possible), 6) if total_possible else None,
        "candidate_file": os.path.abspath(args.output),
    }
    if args.evaluate_ground_truth:
        report["ground_truth_evaluation"] = evaluate_recall(
            args.output, os.path.join(split_dir, "train_ground_truth.tsv"))
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
