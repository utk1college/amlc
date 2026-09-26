#!/usr/bin/env python3
"""Blocking lab: choose, confirm and diagnose the candidate configuration.

Held-out entity sets come from split.lab_set and never include validation:
  block_dev      every selection below
  block_confirm  one accept/reject of the selected configuration (gate 1)
  e2e_confirm    exported for the end-to-end gate only (e2e_gate.py)

Commands
  choose-w   pick the C1 weight on block_dev at baseline depths (4 evaluations)
  select     greedy expansion on block_dev over fixed grids, then text form
  confirm    gate 1 on block_confirm against the v2 candidate file
  oracle     what share of block_dev misses each remaining idea could reach
  export     candidate TSV of one lab set for a configuration (or for v2)

Every configuration is scored by the blocking ceiling (macro F-0.5 of a
perfect matcher restricted to the candidates), link recall and candidates per
entity. Differences are judged against paired-bootstrap standard errors over
entities; the number of evaluations is logged.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import polars as pl

from merge_candidates import censor, combine, reverse_pairs, selection
from split import lab_set
from text_normalize import RE_ALIAS, RE_DOTTED, join_dotted, script_flags, to_ascii

FORWARD_GRID = (0, 10, 20, 30, 50, 100)
REVERSE_GRID = (0, 1, 3, 5, 10)
EMBEDDING_GRID = (0, 5, 10)
W_GRID = (0.4, 0.5, 0.6, 0.7)
BOOTSTRAP = 200


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def w_column(prefix, w):
    return f"{prefix}combined_{round(w * 100):03d}"


# ---------------------------------------------------------------- data

def lab_entities(source1_path):
    rows = []
    with open(source1_path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            name = lab_set(parts[0])
            if name:
                rows.append((parts[0], name, parts[3].strip().casefold()))
    return pl.DataFrame(rows, schema=["source1_id", "lab_set", "country"], orient="row")


def read_truth(path, entities):
    wanted = set(entities["source1_id"].to_list())
    pairs = []
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            source1_id, _, matches = line.rstrip("\n").partition("\t")
            if source1_id in wanted:
                pairs.extend((source1_id, t) for t in matches.split(",") if t)
    return pl.DataFrame(pairs, schema=["source1_id", "target_id"], orient="row")


def read_records(paths, ids):
    records = {}
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            handle.readline()
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if parts[0] in ids:
                    records[parts[0]] = (parts[1], parts[2], parts[3].strip().casefold())
    return records


def link_slices(truth, entities, records):
    """Label each true link with marginal slices: country, source, script, empty address, alias."""
    rows = []
    for target_id in truth["target_id"].unique().to_list():
        name, address, _ = records.get(target_id, ("", "", ""))
        nonlatin, accented = script_flags(name)
        rows.append((target_id, "nonlatin" if nonlatin else "accented" if accented else "latin",
                     not address.strip(), bool(RE_ALIAS.search(join_dotted(to_ascii(name))))))
    attributes = pl.DataFrame(rows, schema=["target_id", "script", "empty_address", "alias"], orient="row")
    return (truth.join(entities.select("source1_id", "country"), on="source1_id")
            .join(attributes, on="target_id", how="left")
            .with_columns(pl.col("target_id").str.slice(0, 2).alias("source")))


def scan_pairs(directory, lab_ids, transform=None):
    frames = []
    for path in sorted(glob.glob(os.path.join(directory, "*.parquet"))):
        frame = pl.read_parquet(path)
        if transform is not None:
            frame = transform(frame)
        frames.append(frame.join(lab_ids, on="source1_id", how="semi"))
    return pl.concat(frames, how="diagonal_relaxed") if frames else None


def load_pairs(args, entities, truth):
    lab_ids = entities.select("source1_id")
    frames = [scan_pairs(args.forward, lab_ids)]
    if getattr(args, "reverse", None):
        frames.append(scan_pairs(args.reverse, lab_ids, reverse_pairs))
    if getattr(args, "embedding", None):
        frames.append(scan_pairs(args.embedding, lab_ids).select("source1_id", "target_id", "emb_rank"))
    if getattr(args, "embedding_alt", None):
        frames.append(scan_pairs(args.embedding_alt, lab_ids).select(
            "source1_id", "target_id", pl.col("emb_rank").alias("emb_alt_rank")))
    pairs = combine(frames)
    return pairs.join(truth.with_columns(pl.lit(True).alias("is_true")), on=["source1_id", "target_id"],
                      how="left").with_columns(pl.col("is_true").fill_null(False))


def read_candidate_tsv(path, wanted):
    pairs = []
    with open(path, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            source1_id, _, ids = line.rstrip("\n").partition("\t")
            if source1_id in wanted:
                pairs.extend((source1_id, t) for t in ids.split(",") if t)
    return pl.DataFrame(pairs, schema=["source1_id", "target_id"], orient="row")


# ---------------------------------------------------------------- metrics

class LabSet:
    """One lab set: its entities, true links and candidate pairs."""

    def __init__(self, name, entities, truth, pairs, slices):
        self.name = name
        self.entities = entities.filter(pl.col("lab_set") == name).select("source1_id").with_row_index("e")
        counts = truth.group_by("source1_id").agg(pl.len().alias("t"))
        self.t = (self.entities.join(counts, on="source1_id", how="left").fill_null(0).sort("e")["t"]
                  .to_numpy().astype(np.float64))
        self.pairs = pairs.join(self.entities, on="source1_id") if pairs is not None else None
        self.slices = slices.join(self.entities, on="source1_id")
        self.boot = np.random.default_rng(0).integers(0, len(self.t), (BOOTSTRAP, len(self.t)))
        self.evaluations = 0

    def per_entity(self, selected):
        n, r = np.zeros(len(self.t)), np.zeros(len(self.t))
        per = selected.group_by("e").agg(pl.len().alias("n"), pl.col("is_true").sum().alias("r"))
        n[per["e"].to_numpy()] = per["n"].to_numpy()
        r[per["e"].to_numpy()] = per["r"].to_numpy()
        recall = np.divide(r, self.t, out=np.zeros_like(r), where=self.t > 0)
        ceiling = np.where(self.t == 0, 1.0, np.where(r > 0, 1.25 * recall / (0.25 + recall), 0.0))
        return {"n": n, "r": r, "ceiling": ceiling, "selected": selected}

    def evaluate(self, config):
        self.evaluations += 1
        return self.per_entity(selection(self.pairs, config).select("e", "target_id", "is_true"))

    def evaluate_pairs(self, pairs):
        """Metrics for an explicit pair list (e.g. the v2 candidate file)."""
        selected = (pairs.join(self.entities, on="source1_id")
                    .join(self.slices.select("source1_id", "target_id").with_columns(pl.lit(True).alias("is_true")),
                          on=["source1_id", "target_id"], how="left").with_columns(pl.col("is_true").fill_null(False)))
        return self.per_entity(selected.select("e", "target_id", "is_true"))

    def summary(self, result):
        n = result["n"]
        return {"entities": len(n), "ceiling_f05": round(float(result["ceiling"].mean()), 6),
                "link_recall": round(float(result["r"].sum() / max(self.t.sum(), 1)), 6),
                "mean_candidates": round(float(n.mean()), 2), "p99_candidates": float(np.percentile(n, 99)),
                "max_candidates": float(n.max())}

    def paired(self, a, b):
        """Gain b - a in ceiling and recall with paired-bootstrap standard errors."""
        diff = b["ceiling"] - a["ceiling"]
        recall_diff = b["r"] - a["r"]
        boot_ceiling = diff[self.boot].mean(axis=1)
        boot_recall = recall_diff[self.boot].sum(axis=1) / np.maximum(self.t[self.boot].sum(axis=1), 1)
        return {"ceiling_gain": float(diff.mean()), "ceiling_se": float(boot_ceiling.std()),
                "recall_gain": float(recall_diff.sum() / max(self.t.sum(), 1)), "recall_se": float(boot_recall.std()),
                "candidate_gain": float((b["n"] - a["n"]).mean())}

    def slice_gains(self, a, b):
        """Recall gain per marginal slice with its paired-bootstrap SE."""
        found = {}
        for label, result in (("a", a), ("b", b)):
            found[label] = self.slices.join(result["selected"].select("e", "target_id"), on=["e", "target_id"], how="semi")
        out = {}
        for column in ("country", "source", "script", "empty_address", "alias"):
            for value in self.slices[column].unique().to_list():
                links = self.slices.filter(pl.col(column) == value)
                total = np.bincount(links["e"].to_numpy(), minlength=len(self.t)).astype(np.float64)
                counts = [np.bincount(found[k].filter(pl.col(column) == value)["e"].to_numpy(),
                                      minlength=len(self.t)).astype(np.float64) for k in ("a", "b")]
                diff = counts[1] - counts[0]
                boot = diff[self.boot].sum(axis=1) / np.maximum(total[self.boot].sum(axis=1), 1)
                out[f"{column}={value}"] = {"links": int(total.sum()), "recall_a": round(counts[0].sum() / max(total.sum(), 1), 5),
                                            "recall_gain": round(diff.sum() / max(total.sum(), 1), 5),
                                            "se": round(float(boot.std()), 5)}
        return out


# ---------------------------------------------------------------- commands

def baseline_config(w):
    return {"rank_name": 20, "rank_address": 20, w_column("rank_", w): 30}


def choose_w(lab):
    results = []
    for w in W_GRID:
        results.append({"w": w, **lab.summary(lab.evaluate(baseline_config(w)))})
        log(f"w={w}: {results[-1]}")
    best = max(results, key=lambda r: r["ceiling_f05"])
    return {"w": best["w"], "grid": results, "evaluations": lab.evaluations}


def settings_for(lab, w):
    columns = set(lab.pairs.columns)
    grid = {"rank_name": FORWARD_GRID, "rank_address": FORWARD_GRID, "rank_address_char": FORWARD_GRID,
            w_column("rank_", w): FORWARD_GRID, w_column("rev_rank_", w): REVERSE_GRID,
            "rev_rank_name": REVERSE_GRID, "rev_rank_address": REVERSE_GRID, "emb_rank": EMBEDDING_GRID}
    return {c: g for c, g in grid.items() if c in columns}


def greedy(lab, w, budget):
    settings = settings_for(lab, w)
    config = {c: 0 for c in settings}
    config.update(baseline_config(w))
    current = lab.evaluate(config)
    history = [{"step": "start", "config": dict(config), **lab.summary(current)}]
    while True:
        moves = []
        for column, values in settings.items():
            larger = [v for v in values if v > config[column]]
            if not larger:
                continue
            trial = {**config, column: larger[0]}
            result = lab.evaluate(trial)
            gain = lab.paired(current, result)
            mean_n = float(result["n"].mean())
            eligible = mean_n <= budget and gain["ceiling_gain"] > 2 * gain["ceiling_se"]
            ratio = gain["ceiling_gain"] / gain["candidate_gain"] if gain["candidate_gain"] > 0 else float("inf")
            moves.append((eligible, ratio, -gain["candidate_gain"], column, larger[0], result, gain, mean_n))
        eligible = [m for m in moves if m[0]]
        if not eligible:
            break
        _, ratio, _, column, value, result, gain, mean_n = max(eligible, key=lambda m: (m[1], m[2]))
        config[column] = value
        current = result
        history.append({"step": f"{column}={value}", "gain": gain, "ratio": ratio, **lab.summary(result)})
        log(f"accepted {column}={value}: {lab.summary(result)}")
    return config, current, history


def choose_text_form(lab, config, current):
    if config.get("emb_rank", 0) <= 0 or "emb_alt_rank" not in lab.pairs.columns:
        return "name", current, None
    primary = lab.summary(current)
    swapped = lab.pairs
    lab.pairs = swapped.with_columns(pl.col("emb_alt_rank").alias("emb_rank"))
    alternative = lab.evaluate(config)
    lab.pairs = swapped
    if alternative["ceiling"].mean() > current["ceiling"].mean():
        return "name_address", alternative, {"name": primary, "name_address": lab.summary(alternative)}
    return "name", current, {"name": primary, "name_address": lab.summary(alternative)}


def oracle(lab, config, result, records_paths, index_dir):
    """Composition of the remaining misses and which ideas could reach them (analysis only)."""
    misses = lab.slices.join(result["selected"].select("e", "target_id"), on=["e", "target_id"], how="anti")
    found = misses.join(lab.pairs, on=["e", "target_id"], how="left")
    forward = [c for c in found.columns if c.startswith("rank_")]
    reverse = [c for c in found.columns if c.startswith("rev_rank_")]
    big = 32767
    report = {"misses": misses.height, "links": int(lab.t.sum())}

    def share(expr):
        return round(float(found.select(expr.cast(pl.Float64).mean()).item() or 0.0), 4)

    if forward:
        report["forward_rank_le_100"] = share(pl.min_horizontal([pl.col(c).fill_null(big) for c in forward]) <= 100)
    if reverse:
        report["reverse_rank_le_10"] = share(pl.min_horizontal([pl.col(c).fill_null(big) for c in reverse]) <= 10)
    if "emb_rank" in found.columns:
        report["embedding_rank_le_10"] = share(pl.col("emb_rank").fill_null(big) <= 10)
    for column in ("country", "source", "script", "empty_address", "alias"):
        report[f"by_{column}"] = dict(misses.group_by(column).len().sort("len", descending=True).iter_rows())

    # Dotted abbreviations and two-hop reachability need the records' text.
    top_other = (lab.pairs.join(misses.select("e").unique(), on="e")
                 .filter(pl.col(w_column("rank_", config["_w"])).fill_null(big) <= 3)
                 .select("e", pl.col("target_id").alias("hop_id")))
    hops = misses.join(top_other, on="e").filter(pl.col("target_id").str.slice(0, 2) != pl.col("hop_id").str.slice(0, 2))
    ids = set(misses["target_id"].to_list()) | set(hops["hop_id"].to_list()) | set(misses["source1_id"].to_list())
    records = read_records(records_paths, ids)
    dotted = [bool(RE_DOTTED.search(to_ascii(records.get(t, ("",))[0])) or RE_DOTTED.search(to_ascii(records.get(s, ("",))[0])))
              for s, t in zip(misses["source1_id"].to_list(), misses["target_id"].to_list())]
    report["dotted_in_either_name"] = round(float(np.mean(dotted)), 4) if dotted else 0.0
    report["two_hop_max_cosine"] = two_hop_cosines(hops, records, index_dir, config["_w"])
    return report


def two_hop_cosines(hops, records, index_dir, w):
    """Distribution of the best C1 cosine between a missed target and the entity's top-3 other-source candidates."""
    if not hops.height:
        return None
    from blocking_knn import featurize, tfidf
    from text_normalize import normalize_address, normalize_name
    values = {}
    for country in {records[t][2] for t in hops["target_id"].to_list() if t in records}:
        idf = np.load(os.path.join(index_dir, country.replace(" ", "_") + ".idf.npy"))
        rows = [(t, h) for t, h in zip(hops["target_id"].to_list(), hops["hop_id"].to_list())
                if t in records and h in records and records[t][2] == country]
        if not rows:
            continue
        vectors = []
        for side in (0, 1):
            ids = [r[side] for r in rows]
            counts = featurize([normalize_name(records[i][0]) for i in ids], [normalize_address(records[i][1]) for i in ids],
                               ("name", "address"))
            vectors.append({v: tfidf(counts[v], idf[j]) for j, v in enumerate(("name", "address"))})
        cosine = (w * np.asarray(vectors[0]["name"].multiply(vectors[1]["name"]).sum(axis=1)).ravel()
                  + (1 - w) * np.asarray(vectors[0]["address"].multiply(vectors[1]["address"]).sum(axis=1)).ravel())
        for (t, _), value in zip(rows, cosine):
            values[t] = max(values.get(t, 0.0), float(value))
    array = np.array(list(values.values()))
    return {"misses_with_hop": len(array),
            "quantiles": {q: round(float(np.quantile(array, q)), 3) for q in (0.5, 0.75, 0.9)} if len(array) else None,
            "share_ge": {x: round(float((array >= x).mean()), 4) for x in (0.5, 0.7, 0.9)} if len(array) else None}


def main():
    parser = argparse.ArgumentParser(description="Blocking lab on held-out training entities.")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", required=True)
    common.add_argument("--out", required=True, help="JSON report (and config) path.")
    for name in ("choose-w", "select", "confirm", "oracle", "export"):
        p = sub.add_parser(name, parents=[common])
        p.add_argument("--forward", help="blocking_knn --lab-only pairs directory")
        p.add_argument("--reverse")
        p.add_argument("--embedding")
        p.add_argument("--embedding-alt", help="Embedding pairs with the alternative text form")
        p.add_argument("--config", help="Configuration JSON from select")
    sub.choices["select"].add_argument("--budget", type=float, required=True, help="Mean candidates per entity (resource bound).")
    sub.choices["select"].add_argument("--w", type=float, required=True, help="C1 weight from choose-w.")
    sub.choices["confirm"].add_argument("--v2", required=True, help="v2 candidate TSV (baseline run)")
    sub.choices["oracle"].add_argument("--index-dir", required=True, help="Forward index (for IDF)")
    sub.choices["export"].add_argument("--set", default="e2e_confirm")
    sub.choices["export"].add_argument("--v2", help="Export this v2 TSV restricted to the set instead")
    sub.choices["export"].add_argument("--tsv", required=True)
    sub.choices["export"].add_argument("--pairs-dir", help="Also write the selected pairs with metadata (Parquet)")
    args = parser.parse_args()

    started = time.time()
    train = os.path.join(args.data_dir, "train")
    entities = lab_entities(os.path.join(train, "train_source1.tsv"))
    truth = read_truth(os.path.join(train, "train_ground_truth.tsv"), entities)
    config = None
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            config = json.load(handle)

    if args.command == "export":
        members = entities.filter(pl.col("lab_set") == args.set)
        if args.v2:
            pairs = read_candidate_tsv(args.v2, set(members["source1_id"].to_list()))
        else:
            pairs = selection(combine([f for f in (
                scan_pairs(args.forward, members.select("source1_id")),
                scan_pairs(args.reverse, members.select("source1_id"), reverse_pairs) if args.reverse else None,
                scan_pairs(args.embedding if config.get("emb_form", "name") == "name" else args.embedding_alt,
                           members.select("source1_id")) if (args.embedding and config.get("emb_rank", 0) > 0) else None)
                if f is not None]), config)
        if not args.v2 and args.pairs_dir:
            os.makedirs(args.pairs_dir, exist_ok=True)
            censor(pairs, config).write_parquet(os.path.join(args.pairs_dir, "part-000.parquet"))
        lists = dict(pairs.group_by("source1_id").agg(pl.col("target_id").sort()).iter_rows())
        with open(args.tsv, "w", encoding="utf-8") as handle:
            handle.write("source1_entity_id\tcandidate_entity_ids\n")
            for source1_id in members["source1_id"].to_list():
                handle.write(f"{source1_id}\t{','.join(lists.get(source1_id, []))}\n")
        report = {"set": args.set, "entities": members.height, "pairs": pairs.height,
                  "mean_candidates": round(pairs.height / max(members.height, 1), 2)}
    else:
        targets = set(truth["target_id"].to_list())
        records_paths = [os.path.join(train, f"train_source{n}.tsv") for n in (2, 3)]
        slices = link_slices(truth, entities, read_records(records_paths, targets))
        pairs = load_pairs(args, entities, truth)
        log(f"loaded {pairs.height} lab pairs")
        if args.command == "choose-w":
            report = choose_w(LabSet("block_dev", entities, truth, pairs, slices))
        elif args.command == "select":
            lab = LabSet("block_dev", entities, truth, pairs, slices)
            chosen, current, history = greedy(lab, args.w, args.budget)
            form, current, forms = choose_text_form(lab, chosen, current)
            config = {**chosen, "emb_form": form, "_w": args.w}
            report = {"config": config, "dev": lab.summary(current), "budget": args.budget,
                      "evaluations": lab.evaluations, "text_forms": forms, "history": history}
            with open(os.path.splitext(args.out)[0] + ".config.json", "w", encoding="utf-8") as handle:
                json.dump(config, handle, indent=2)
        elif args.command == "confirm":
            v2 = read_candidate_tsv(args.v2, set(entities["source1_id"].to_list()))
            report = {"config": config}
            gains = {}
            for name in ("block_dev", "block_confirm"):
                lab = LabSet(name, entities, truth, pairs, slices)
                a, b = lab.evaluate_pairs(v2), lab.evaluate(config)
                gains[name] = lab.paired(a, b)
                report[name] = {"v2": lab.summary(a), "v3": lab.summary(b), "gain": gains[name]}
                if name == "block_confirm":
                    report["slices"] = lab.slice_gains(a, b)
            confirm, dev = gains["block_confirm"], gains["block_dev"]
            gap_se = (confirm["ceiling_se"] ** 2 + dev["ceiling_se"] ** 2) ** 0.5
            worst = min(report["slices"].items(), key=lambda kv: kv[1]["recall_gain"] / max(kv[1]["se"], 1e-9))
            report["gate1"] = {
                "ceiling_beats_v2": confirm["ceiling_gain"] > 2 * confirm["ceiling_se"],
                "recall_beats_v2": confirm["recall_gain"] > 2 * confirm["recall_se"],
                "dev_confirm_gap_ok": abs(dev["ceiling_gain"] - confirm["ceiling_gain"]) <= 2 * gap_se,
                "no_slice_regression": all(v["recall_gain"] >= -2 * v["se"] for v in report["slices"].values()),
                "worst_slice": worst[0]}
            report["gate1"]["pass"] = all(v for k, v in report["gate1"].items() if k != "worst_slice")
        else:  # oracle
            lab = LabSet("block_dev", entities, truth, pairs, slices)
            report = oracle(lab, config, lab.evaluate(config),
                            [os.path.join(train, "train_source1.tsv")] + records_paths, args.index_dir)
    report["seconds"] = round(time.time() - started, 1)
    text = json.dumps(report, indent=2, default=str)
    print(text)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")


if __name__ == "__main__":
    main()
