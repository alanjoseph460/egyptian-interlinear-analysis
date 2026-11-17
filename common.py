"""Shared by every experiment: data loading, validation folds, scoring and saving predictions."""
import csv
import json
import os
import re

import numpy as np

import metric

DATA = "dataset/public"


def load(name):
    with open(f"{DATA}/{name}.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["words"] = r["hieroglyphs"].split(" ")
        r["century"] = int(re.match(r"\d+", r["latest_century"]).group())
        if "prediction" in r:
            r["groups"] = r["prediction"].split(" ")
    return rows


def fold_ids(rows):
    """0 = 18th century BCE and later, 1 = 19th, 2 = 20th to 22nd, -1 = earlier (never held out)."""
    out = np.full(len(rows), -1)
    for i, r in enumerate(rows):
        c = r["century"]
        out[i] = 0 if c <= 18 else 1 if c == 19 else 2 if c <= 22 else -1
    return out


def held_mask(rows, mode):
    """page: 19th century or later held out (the split the problem text names, 1,809 rows, 46% unseen
    spellings, harder than test). near: 18th or later (1,499 rows, 38% unseen, closest to the test
    mix). f2: 20th to 22nd (2,500 rows, 24% unseen). none: nothing held out."""
    f = fold_ids(rows)
    return {"page": (f == 0) | (f == 1), "near": f == 0, "f1": f == 1, "f2": f == 2,
            "none": np.zeros(len(rows), bool)}[mode]


def score_rows(pred, rows):
    """pred: one analysis string per row. Returns dict with s, T, L, P, G means."""
    return metric.score_parts([r["prediction"] for r in rows], list(pred))


def save_preds(out_dir, name, ids, preds, scores=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{name}.json", "w", encoding="utf-8") as f:
        json.dump({"ids": list(ids), "pred": list(preds), "scores": scores}, f, ensure_ascii=False)
