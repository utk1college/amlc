#!/usr/bin/env python3
"""Embedding retrieval for cross-script names: target -> Source 1, same country.

Trigger: a Source-2/3 record whose name contains non-Latin letters (a
country-agnostic rule; in this data that is the Indic-script slice). Each
triggered record queries the Source-1 records of its own country label.

Model: intfloat/multilingual-e5-small (MIT), loaded from a local folder.
Texts get the model card's "query: " prefix on both sides (symmetric
similarity), are mean-pooled and L2-normalized, so inner product = cosine.

Search: FAISS IndexHNSWFlat with inner-product metric, M=32 and
efConstruction=40 (FAISS defaults); efSearch is a parameter chosen by the
`tune` command, which measures ANN recall against exact IndexFlatIP on the
block_dev links only.

Commands
  encode  embed Source 1 and the triggered targets of a split (cached .npy)
  tune    ANN vs exact recall@k on block_dev for a range of efSearch
  search  top-k Source-1 records per triggered target -> Parquet
          (target_id, source1_id, emb_rank, emb_score)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import polars as pl

from split import lab_set
from text_normalize import script_flags

M, EF_CONSTRUCTION = 32, 40  # FAISS defaults


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def text_of(name, address, form):
    return f"query: {name}" if form == "name" else f"query: {name}, {address}"


def read_side(paths, trigger):
    """(ids, countries, names, addresses) of records, optionally only the triggered ones."""
    ids, countries, names, addresses = [], [], [], []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                entity_id, name, address, country = line.rstrip("\n").split("\t")
                if trigger and not script_flags(name)[0]:
                    continue
                ids.append(entity_id)
                countries.append(country.strip().casefold())
                names.append(name)
                addresses.append(address)
    return ids, countries, names, addresses


class Encoder:
    def __init__(self, model_dir, device, batch_size, threads):
        import torch
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(threads)
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(device).eval()
        self.batch_size = batch_size

    def __call__(self, texts):
        out = np.zeros((len(texts), self.model.config.hidden_size), np.float32)
        order = np.argsort([len(t) for t in texts])  # similar lengths per batch -> less padding
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                index = order[start:start + self.batch_size]
                batch = self.tokenizer([texts[i] for i in index], padding=True, truncation=True,
                                       max_length=64, return_tensors="pt").to(self.device)
                hidden = self.model(**batch).last_hidden_state
                mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
                out[index] = pooled.float().cpu().numpy()
                if start // self.batch_size % 200 == 0:
                    log(f"encoded {start + len(index)}/{len(texts)}")
        return out


def encode(args):
    split_dir = os.path.join(args.data_dir, args.split)
    encoder = Encoder(args.model_dir, args.device, args.batch_size, args.threads)
    os.makedirs(args.out_dir, exist_ok=True)
    report = {}
    for side, paths, trigger in (("source1", [os.path.join(split_dir, f"{args.split}_source1.tsv")], False),
                                 ("targets", [os.path.join(split_dir, f"{args.split}_source{n}.tsv") for n in (2, 3)], True)):
        ids, countries, names, addresses = read_side(paths, trigger)
        if side == "source1":
            # Only countries that have triggered targets need a Source-1 index.
            wanted = set(read_side([os.path.join(split_dir, f"{args.split}_source{n}.tsv") for n in (2, 3)], True)[1])
            keep = [i for i, c in enumerate(countries) if c in wanted]
            ids, countries, names, addresses = ([x[i] for i in keep] for x in (ids, countries, names, addresses))
        started = time.time()
        vectors = encoder([text_of(n, a, args.form) for n, a in zip(names, addresses)])
        np.save(os.path.join(args.out_dir, f"{side}.{args.form}.npy"), vectors)
        pl.DataFrame({"id": ids, "country": countries}).write_parquet(os.path.join(args.out_dir, f"{side}.ids.parquet"))
        norms = np.linalg.norm(vectors, axis=1)
        report[side] = {"records": len(ids), "seconds": round(time.time() - started, 1),
                        "norm_min": float(norms.min()) if len(norms) else None, "norm_max": float(norms.max()) if len(norms) else None}
        log(f"{side}: {report[side]}")
    return report


def load(out_dir, side, form):
    ids = pl.read_parquet(os.path.join(out_dir, f"{side}.ids.parquet"))
    return ids, np.load(os.path.join(out_dir, f"{side}.{form}.npy"), mmap_mode="r")


def hnsw(vectors, threads):
    import faiss
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexHNSWFlat(vectors.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def exact(vectors):
    import faiss
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def by_country(ids):
    return {c: np.flatnonzero(ids["country"].to_numpy() == c) for c in ids["country"].unique().to_list()}


def tune(args):
    """ANN recall@k vs exact recall@k over block_dev links whose target is triggered."""
    truth = {}
    with open(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"), encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            source1_id, _, matches = line.rstrip("\n").partition("\t")
            if lab_set(source1_id) == "block_dev":
                for target in matches.split(","):
                    if target:
                        truth[target] = source1_id
    s1_ids, s1_vectors = load(args.out_dir, "source1", args.form)
    t_ids, t_vectors = load(args.out_dir, "targets", args.form)
    s1_countries, report = by_country(s1_ids), {"k": args.k, "ef_search": {}}
    queries = [i for i, t in enumerate(t_ids["id"].to_list()) if t in truth]
    rng = np.random.default_rng(0)
    boot = None
    hits = {"exact": [], **{ef: [] for ef in args.ef_grid}}
    for country, members in s1_countries.items():
        rows = [i for i in queries if t_ids["country"][i] == country]
        if not rows:
            continue
        vectors = np.asarray(s1_vectors[members])
        member_ids = np.array(s1_ids["id"].to_list())[members]
        wanted = [truth[t_ids["id"][i]] for i in rows]
        q = np.asarray(t_vectors[rows])
        _, found = exact(vectors).search(q, args.k)
        hits["exact"].extend(w in set(member_ids[f]) for w, f in zip(wanted, found))
        index = hnsw(vectors, args.threads)
        for ef in args.ef_grid:
            index.hnsw.efSearch = ef
            started = time.time()
            _, found = index.search(q, args.k)
            hits[ef].extend(w in set(member_ids[f]) for w, f in zip(wanted, found))
            report["ef_search"].setdefault(str(ef), {})[f"{country}_query_ms"] = round(1000 * (time.time() - started) / len(rows), 3)
    reference = np.array(hits["exact"], dtype=float)
    boot = rng.integers(0, len(reference), (200, len(reference)))
    report["links"] = len(reference)
    report["exact_recall"] = float(reference.mean())
    chosen = None
    for ef in args.ef_grid:
        values = np.array(hits[ef], dtype=float)
        se = float((reference - values)[boot].mean(axis=1).std())
        report["ef_search"][str(ef)].update(recall=float(values.mean()), gap_to_exact=float(reference.mean() - values.mean()), se=se)
        if chosen is None and reference.mean() - values.mean() <= se:
            chosen = ef
    report["chosen_ef_search"] = chosen
    return report


def search(args):
    s1_ids, s1_vectors = load(args.out_dir, "source1", args.form)
    t_ids, t_vectors = load(args.out_dir, "targets", args.form)
    s1_countries = by_country(s1_ids)
    t_countries = by_country(t_ids)
    os.makedirs(args.pairs_out, exist_ok=True)
    report, part = {"k": args.k, "ef_search": args.ef_search, "countries": {}}, 0
    for country, members in s1_countries.items():
        rows = t_countries.get(country)
        if rows is None or not len(rows):
            continue
        started = time.time()
        index = hnsw(np.asarray(s1_vectors[members]), args.threads)
        index.hnsw.efSearch = args.ef_search
        member_ids = pl.Series(np.array(s1_ids["id"].to_list())[members])
        target_ids = t_ids["id"].to_numpy()
        for start in range(0, len(rows), 200_000):
            block = rows[start:start + 200_000]
            scores, found = index.search(np.asarray(t_vectors[block]), args.k)
            r, c = np.nonzero(found >= 0)
            pl.DataFrame({"target_id": target_ids[block][r], "source1_id": member_ids.gather(found[r, c]),
                          "emb_rank": (c + 1).astype(np.int16), "emb_score": scores[r, c].astype(np.float32)}
                         ).write_parquet(os.path.join(args.pairs_out, f"part-{part:05d}.parquet"))
            part += 1
        report["countries"][country] = {"source1": len(members), "targets": len(rows), "seconds": round(time.time() - started, 1)}
        log(f"{country}: {report['countries'][country]}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Embedding retrieval for non-Latin-script targets.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("encode", "tune", "search"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", required=True)
        p.add_argument("--out-dir", required=True, help="Embedding cache directory for one split.")
        p.add_argument("--form", choices=("name", "name_address"), default="name")
        p.add_argument("--threads", type=int, default=int(os.environ.get("WORKERS", os.cpu_count() or 4)))
        p.add_argument("--report")
    sub.choices["encode"].add_argument("--split", choices=("train", "test"), required=True)
    sub.choices["encode"].add_argument("--model-dir", required=True, help="Local folder of intfloat/multilingual-e5-small")
    sub.choices["encode"].add_argument("--device", default="cpu")
    sub.choices["encode"].add_argument("--batch-size", type=int, default=256)
    sub.choices["tune"].add_argument("--k", type=int, default=10)
    sub.choices["tune"].add_argument("--ef-grid", type=lambda s: [int(x) for x in s.split(",")], default=[16, 64, 128, 256, 512])
    sub.choices["search"].add_argument("--k", type=int, default=10)
    sub.choices["search"].add_argument("--ef-search", type=int, required=True)
    sub.choices["search"].add_argument("--pairs-out", required=True)
    args = parser.parse_args()
    started = time.time()
    report = {"encode": encode, "tune": tune, "search": search}[args.command](args)
    report["seconds"] = round(time.time() - started, 1)
    text = json.dumps(report, indent=2)
    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
