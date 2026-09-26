#!/usr/bin/env python3
"""On-disk store of normalized records, so later stages never load a whole source.

Each record of Sources 1-3 of a split is normalized once (name, name skeleton,
address, address skeleton) and written to SQLite in streamed batches. Stages
that need the text of a batch of records fetch exactly those IDs.
"""

import argparse
import os
import sqlite3
import sys
import time
from collections import deque
from multiprocessing import get_all_start_methods, get_context

from text_normalize import core_name, name_variants, normalize_address, script_flags, skeleton

# variants: the alias parts of an "X aka Y" name joined by "|" (empty when none).
COLUMNS = ("id", "country", "name", "name_skeleton", "address", "address_skeleton", "core", "variants",
           "nonlatin", "accented")
BATCH_ROWS = 100_000


def normalize_rows(rows):
    out = []
    for entity_id, name, address, country in rows:
        variants = name_variants(name)
        normalized_address = normalize_address(address)
        nonlatin, accented = script_flags(name)
        out.append((entity_id, country.strip().casefold(), variants[0], skeleton(variants[0]),
                    normalized_address, skeleton(normalized_address), core_name(name), "|".join(variants[1:]),
                    int(nonlatin), int(accented)))
    return out


def stream_rows(path, size):
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        batch = []
        for line in handle:
            batch.append(tuple(line.rstrip("\n").split("\t")))
            if len(batch) >= size:
                yield batch
                batch = []
        if batch:
            yield batch


def bounded_imap(pool, function, items, limit):
    """Ordered pool.imap that never holds more than `limit` unfinished items."""
    pending = deque()
    for item in items:
        pending.append(pool.apply_async(function, (item,)))
        if len(pending) >= limit:
            yield pending.popleft().get()
    while pending:
        yield pending.popleft().get()


def build(data_dir, split, database, workers):
    if os.path.exists(database):
        os.remove(database)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute(f"CREATE TABLE records ({', '.join(COLUMNS)})")
    insert = f"INSERT INTO records VALUES ({', '.join('?' * len(COLUMNS))})"
    total = 0
    with get_context("fork" if "fork" in get_all_start_methods() else "spawn").Pool(workers) as pool:
        for source in (1, 2, 3):
            path = os.path.join(data_dir, split, f"{split}_source{source}.tsv")
            for rows in bounded_imap(pool, normalize_rows, stream_rows(path, BATCH_ROWS), workers + 2):
                connection.executemany(insert, rows)
                total += len(rows)
            print(f"[{time.strftime('%H:%M:%S')}] source {source} stored ({total} rows)", file=sys.stderr, flush=True)
    connection.execute("CREATE UNIQUE INDEX records_id ON records(id)")
    connection.commit()
    connection.close()
    return total


class RecordStore:
    """Fetch normalized records by ID from a store built by this module."""

    def __init__(self, database):
        self.connection = sqlite3.connect(database)
        self.connection.execute("PRAGMA temp_store = MEMORY")
        # Random lookups are disk-bound until the file is in the OS page cache;
        # memory-mapping lets cached pages be read without copying.
        self.connection.execute("PRAGMA mmap_size = 4000000000")

    @staticmethod
    def warm(database):
        """Read the file once so later random lookups hit the page cache."""
        with open(database, "rb") as handle:
            while handle.read(64 * 1024 * 1024):
                pass

    def fetch(self, ids):
        cursor = self.connection.cursor()
        cursor.execute("DROP TABLE IF EXISTS temp.wanted")
        cursor.execute("CREATE TEMP TABLE wanted (id TEXT PRIMARY KEY)")
        cursor.executemany("INSERT OR IGNORE INTO temp.wanted VALUES (?)", ((i,) for i in sorted(ids)))
        cursor.execute(f"SELECT {', '.join('r.' + c for c in COLUMNS)} FROM temp.wanted w "
                       "JOIN records r ON r.id = w.id")
        return {row[0]: row for row in cursor.fetchall()}

    def close(self):
        self.connection.close()


def main():
    parser = argparse.ArgumentParser(description="Build the normalized record store for one split.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 4)))
    args = parser.parse_args()
    started = time.time()
    total = build(args.data_dir, args.split, args.database, args.workers)
    print(f"{total} records stored in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
