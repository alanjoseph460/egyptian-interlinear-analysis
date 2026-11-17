import sys, re, collections
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = "solutions/working/egyptian-interlinear-analysis/dataset/public/"
tr = pd.read_csv(D + "train.csv", encoding="utf-8", keep_default_na=False)
te = pd.read_csv(D + "test.csv", encoding="utf-8", keep_default_na=False)
C = collections.Counter
tr["cn"] = tr.latest_century.str.extract(r"^(\d+)")[0].astype(int)
te["cn"] = te.latest_century.str.extract(r"^(\d+)")[0].astype(int)
rows = []
for i, (h, p) in enumerate(zip(tr.hieroglyphs, tr.prediction)):
    ws = h.split(" ")
    for j, (w, gr) in enumerate(zip(ws, p.split(" "))):
        f = gr.split(";")
        rows.append((i, j, w, f[0], f[1], f[2], f[3], gr, ws[j - 1] if j else "<s>", ws[j + 1] if j + 1 < len(ws) else "</s>"))
W = pd.DataFrame(rows, columns=["row", "pos", "w", "tl", "lem", "upos", "gloss", "full", "prev", "next"])
W["cn"] = tr.cn.values[W.row.values]


def maj(df, key, col, minc=1):
    keys = key if isinstance(key, list) else [key]
    c = df.groupby(keys + [col]).size().reset_index(name="n")
    g = c.groupby(keys).n.max()
    n = c.groupby(keys).n.sum()
    k = n >= minc
    return g[k].sum() / n[k].sum(), int(n[k].sum()), int(k.sum())


print("== 5 ambiguity: majority share by spelling (token-weighted)")
for minc in (1, 2, 5, 20):
    out = []
    for col in ("full", "tl", "lem", "upos", "gloss"):
        s, n, k = maj(W, "w", col, minc)
        out.append(f"{col} {s:.4f}")
    print(f"  spellings seen >={minc}: tokens {n} ({n/len(W):.3f} of all) spellings {k} | " + " ".join(out))
# with one neighbour as context
for key, nm in ((["prev", "w"], "prev+w"), (["w", "next"], "w+next"), (["prev", "w", "next"], "prev+w+next")):
    cnt5 = W.groupby("w").w.transform("size") >= 5
    s, n, k = maj(W[cnt5], key, "full")
    print(f"  spellings >=5, majority full analysis given {nm}: {s:.4f} (in-sample, upper bound)")
n5 = W.groupby("w").w.transform("size") >= 5
amb = W[n5].groupby("w").full.nunique()
print("  spellings >=5 with >1 full analysis:", int((amb > 1).sum()), "of", len(amb), "| tokens in them",
      int(W[n5].w.isin(amb[amb > 1].index).sum()), "of", int(n5.sum()))
# most ambiguous frequent spellings
c5 = W[n5].groupby(["w", "full"]).size().reset_index(name="n")
g = c5.groupby("w").n.agg(["max", "sum", "size"])
g["loss"] = g["sum"] - g["max"]
for k, v in g.sort_values("loss", ascending=False).head(12).iterrows():
    top = W[W.w == k].full.value_counts().head(4).to_dict()
    print(f"    {k} n={v['sum']} maj={v['max']/v['sum']:.2f} distinct={v['size']} {top}")

print("\n== 6 transliteration vs lemma")
tl, lem = W.tl, W.lem
np_ = tl.str.replace(r"[()]", "", regex=True)
e0 = tl == lem
e1 = np_ == lem
cut = np_.str.replace(r"(?<=.)\..*$", "", regex=True)
e2 = cut == lem
e2i = cut.str.lower() == lem.str.lower()
e1i = np_.str.lower() == lem.str.lower()
pre = pd.Series([b.lower().startswith(a.lower()) or a.lower().startswith(b.lower()) for a, b in zip(lem, np_)])
print(f"  lemma == translit {e0.mean():.4f} | caseless {(tl.str.lower()==lem.str.lower()).mean():.4f}")
print(f"  lemma == translit without ( ) {e1.mean():.4f} | caseless {e1i.mean():.4f}")
print(f"  lemma == translit without ( ) and cut at first '.' {e2.mean():.4f} | caseless {e2i.mean():.4f}")
print(f"  any of the three {(e0|e1|e2).mean():.4f} | caseless {(e1i|e2i).mean():.4f} | one is a caseless prefix of the other {pre.mean():.4f}")
print("  by UPOS: any-of-three caseless", ((e1i | e2i).groupby(W.upos).mean()).round(3).to_dict())
print("  lemma has uppercase:", lem.str.contains(r"[A-ZŠḎḤḪṮꜢꜤꞼ]").mean().round(4), "by UPOS", lem.str.contains(r"[A-ZŠḎḤḪṮꜢꜤꞼ]").groupby(W.upos).mean().round(3).to_dict())
print("  translit has uppercase:", tl.str.contains(r"[A-ZŠḎḤḪṮꜢꜤꞼ]").mean().round(4))
print("  translit has '(':", tl.str.contains(r"\(").mean().round(4), "has '.':", tl.str.contains(r"\.").mean().round(4),
      "starts '=':", tl.str.startswith("=").mean().round(4), "has '-':", tl.str.contains("-").mean().round(4), "has i̯:", tl.str.contains("i̯").mean().round(4))
print("  examples not matching:", W[~(e1i | e2i)].sample(14, random_state=0)[["w", "tl", "lem", "upos", "gloss"]].values.tolist())
for key in ("lem", "tl", "w"):
    for col in ("upos", "gloss"):
        s, n, k = maj(W, key, col)
        print(f"  majority {col} given {key}: {s:.4f}")
s, _, _ = maj(W, ["lem", "upos"], "gloss"); print(f"  majority gloss given lemma+upos: {s:.4f}")
s, _, _ = maj(W, ["tl", "lem"], "gloss"); print(f"  majority gloss given translit+lemma: {s:.4f}")
s, _, _ = maj(W, ["tl", "lem"], "upos"); print(f"  majority upos given translit+lemma: {s:.4f}")
s, _, _ = maj(W, "gloss", "upos"); print(f"  majority upos given gloss: {s:.4f}")
s, _, _ = maj(W, "tl", "lem"); print(f"  majority lemma given translit: {s:.4f}")
s, _, _ = maj(W, "lem", "tl"); print(f"  majority translit given lemma: {s:.4f}")
print("  gloss by UPOS: distinct", W.groupby("upos").gloss.nunique().to_dict())

print("\n== 7 leakage checks")
print("id pattern ok train", tr.id.str.fullmatch(r"S[0-9a-f]{10}").mean(), "test", te.id.str.fullmatch(r"S[0-9a-f]{10}").mean(),
      "id lengths", tr.id.str.len().value_counts().to_dict())
print("file sorted by id: train", bool((tr.id.values[:-1] < tr.id.values[1:]).all()), "test", bool((te.id.values[:-1] < te.id.values[1:]).all()))
iv = tr.id.str[1:].map(lambda x: int(x, 16)).values.astype(float)
from scipy.stats import spearmanr, ks_2samp
print("spearman id value vs century train", spearmanr(iv, tr.cn.values), "vs n_words", spearmanr(iv, tr.n_words.values))
tiv = te.id.str[1:].map(lambda x: int(x, 16)).values.astype(float)
print("spearman id value vs century test", spearmanr(tiv, te.cn.values))
print("id value range train", iv.min() / 16**10, iv.max() / 16**10, "test", tiv.min() / 16**10, tiv.max() / 16**10, "KS train vs test id", ks_2samp(iv, tiv))
print("id first hex digit train", C(tr.id.str[1]).most_common(), )
print("id last hex digit train", C(tr.id.str[-1]).most_common())
print("late slice share by first hex digit", tr.groupby(tr.id.str[1]).cn.apply(lambda s: (s <= 19).mean()).round(3).to_dict())
same = (tr.cn.values[:-1] == tr.cn.values[1:]).mean()
p = tr.cn.value_counts(normalize=True).values
print("adjacent rows same century", round(float(same), 4), "expected if shuffled", round(float((p ** 2).sum()), 4))
late = (tr.cn.values <= 19)
print("adjacent rows both late", round(float((late[:-1] & late[1:]).mean()), 4), "expected", round(float(late.mean() ** 2), 4))
# do adjacent rows share words more than random pairs (same text clue)
sets = [set(s.split(" ")) for s in tr.hieroglyphs]
jac = lambda a, b: len(a & b) / len(a | b)
adj = np.mean([jac(sets[i], sets[i + 1]) for i in range(len(sets) - 1)])
rng = np.random.RandomState(0)
perm = rng.permutation(len(sets))
rnd = np.mean([jac(sets[perm[i]], sets[perm[i + 1]]) for i in range(len(sets) - 1)])
print("word Jaccard adjacent rows", round(float(adj), 4), "random pairs", round(float(rnd), 4))
tsets = [set(s.split(" ")) for s in te.hieroglyphs]
adj = np.mean([jac(tsets[i], tsets[i + 1]) for i in range(len(tsets) - 1)])
perm = rng.permutation(len(tsets))
rnd = np.mean([jac(tsets[perm[i]], tsets[perm[i + 1]]) for i in range(len(tsets) - 1)])
print("test: word Jaccard adjacent rows", round(float(adj), 4), "random pairs", round(float(rnd), 4),
      "adjacent same century", round(float((te.cn.values[:-1] == te.cn.values[1:]).mean()), 4), "expected", round(float((te.cn.value_counts(normalize=True).values ** 2).sum()), 4))
# is id a function of the text? duplicates with different ids
vc = tr.hieroglyphs.value_counts()
k = vc.index[0]
print("ids of the most repeated string", tr[tr.hieroglyphs == k].id.head(5).tolist())
# n_words by century as a weak text-type proxy; century is a legit per-row input
print("test n_words by century", te.groupby("cn").n_words.agg(["count", "mean", "median"]).round(2).to_dict("index"))

print("\n== folds")


def fold_ids(train_df):
    cn = train_df["latest_century"].str.extract(r"^(\d+)")[0].astype(int).to_numpy()
    f = np.full(len(train_df), -1, dtype=np.int64)
    f[cn <= 19] = 0
    f[(cn >= 20) & (cn <= 22)] = 1
    f[cn == 23] = 2
    return f


f = fold_ids(tr)
fw = f[W.row.values]
cnw = W.cn.values
for k, trainmask, nm in (
    (0, cnw >= 20, "fold 0 (<=19th) train on >=20th"),
    (1, cnw >= 23, "fold 1 (20-22nd) train on >=23rd only"),
    (1, fw != 1, "fold 1 (20-22nd) train on all other"),
    (2, cnw >= 24, "fold 2 (23rd) train on >=24th only"),
    (2, fw != 2, "fold 2 (23rd) train on all other"),
):
    h = W[fw == k]; r = W[trainmask]
    us = (~h.w.isin(set(r.w))).mean(); ul = (~h.lem.isin(set(r.lem))).mean()
    # lookup diagnostic: majority analysis of the spelling in the training part
    top = (r.groupby(["w", "full"]).size().reset_index(name="n").sort_values("n", ascending=False, kind="stable")
           .drop_duplicates("w").set_index("w").full)
    pred = h.w.map(top)
    seen = pred.notna()
    pf = pred.fillna(";;;").str.split(";", expand=True)
    acc = [(pf[i].values == h[c].values).mean() for i, c in enumerate(("tl", "lem", "upos", "gloss"))]
    accs = [(pf[i].values[seen.values] == h[c].values[seen.values]).mean() for i, c in enumerate(("tl", "lem", "upos", "gloss"))]
    print(f"  {nm}: held rows {int((f==k).sum())} words {len(h)} mean len {len(h)/max(1,int((f==k).sum())):.2f} | train rows {r.row.nunique()} | "
          f"unseen spelling {us:.4f} unseen lemma {ul:.4f} | lookup word acc tl/lem/upos/gloss all {np.round(acc,3).tolist()} seen-only {np.round(accs,3).tolist()}")
print("fold sizes", C(f.tolist()))
