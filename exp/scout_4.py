import sys
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = "solutions/working/egyptian-interlinear-analysis/dataset/public/"
tr = pd.read_csv(D + "train.csv", encoding="utf-8", keep_default_na=False)


def fold_ids(train_df):
    # 0 = 18th century and later, 1 = 19th, 2 = 20th to 22nd, -1 = 23rd and earlier (never held out)
    cn = train_df["latest_century"].str.extract(r"^(\d+)", expand=False).astype(int).to_numpy()
    f = np.full(len(train_df), -1, dtype=np.int64)
    f[cn <= 22] = 2
    f[cn <= 19] = 1
    f[cn <= 18] = 0
    return f


def val_split(train_df, mode="page"):
    # page: folds 0+1 held out (the text's validation). near: fold 0 held out, 19th century kept in training.
    f = fold_ids(train_df)
    held = {"page": f <= 1, "near": f == 0, "f1": f == 1, "f2": f == 2}[mode] & (f >= 0)
    return np.where(~held)[0], np.where(held)[0]


f = fold_ids(tr)
print("fold sizes", {int(k): int((f == k).sum()) for k in np.unique(f)})
print("words per fold", {int(k): int(tr.n_words[f == k].sum()) for k in np.unique(f)})
for mode in ("page", "near", "f1", "f2"):
    a, b = val_split(tr, mode)
    ws = set(w for s in tr.hieroglyphs.values[a] for w in s.split(" "))
    hw = [w for s in tr.hieroglyphs.values[b] for w in s.split(" ")]
    print(mode, "train rows", len(a), "held rows", len(b), "held words", len(hw), "unseen spelling", round(float(np.mean([w not in ws for w in hw])), 4))
tb = tr.prediction.map(lambda s: len(s.encode("utf-8")))
print("train rows with target bytes >512", int((tb > 512).sum()), ">768", int((tb > 768).sum()), ">1024", int((tb > 1024).sum()))
print("target bytes per word incl. separator", round((tb.sum() + len(tr)) / tr.n_words.sum(), 2))
r = (tb / tr.n_words)
print("bytes per word by sentence: p50", round(float(r.median()), 1), "p95", round(float(r.quantile(.95)), 1))
long = tr[tr.n_words >= 20]
print("train rows >=20 words", len(long), "their target bytes p50/p95/max", tb[long.index].median(), tb[long.index].quantile(.95), tb[long.index].max())
