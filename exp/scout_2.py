import sys, re, collections
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = "solutions/working/egyptian-interlinear-analysis/dataset/public/"
tr = pd.read_csv(D + "train.csv", encoding="utf-8", keep_default_na=False)
te = pd.read_csv(D + "test.csv", encoding="utf-8", keep_default_na=False)
C = Counter = collections.Counter
tr["cn"] = tr.latest_century.str.extract(r"^(\d+)")[0].astype(int)
te["cn"] = te.latest_century.str.extract(r"^(\d+)")[0].astype(int)

print("== odd rows: train 16th century")
for r in tr[tr.cn == 16].itertuples():
    print(r.id, r.n_words, r.hieroglyphs, "|", r.prediction[:150])

print("\n== 3 duplicates")
vc = tr.hieroglyphs.value_counts()
dup_rows = int((tr.hieroglyphs.map(vc) > 1).sum())
print("train distinct strings", len(vc), "rows sharing string with another train row", dup_rows, f"{dup_rows/len(tr):.4f}",
      "surplus rows", len(tr) - len(vc))
print("top repeated:", [(k, v) for k, v in vc.head(8).items()])
d = tr[tr.hieroglyphs.map(vc) > 1]
print("dup rows by n_words (1,2,3,4-6,7+):", [(int((d.n_words == 1).sum())), int((d.n_words == 2).sum()), int((d.n_words == 3).sum()),
      int(d.n_words.between(4, 6).sum()), int((d.n_words >= 7).sum())])
g = tr.groupby("hieroglyphs").prediction.nunique()
multi = g[g > 1]
print("dup strings", int((vc > 1).sum()), "of which with >1 distinct prediction", len(multi),
      "rows in such strings", int(tr.hieroglyphs.isin(multi.index).sum()))
# majority-consistency for duplicate rows
num = sum(tr[tr.hieroglyphs == k].prediction.value_counts().iloc[0] for k in vc[vc > 1].index)
print("dup rows: share with the majority prediction of their string", f"{num/dup_rows:.4f}")
g2 = tr.groupby("hieroglyphs").cn.nunique()
print("dup strings spanning >1 century", int((g2[vc[vc > 1].index] > 1).sum()))
trs = set(tr.hieroglyphs)
tvc = te.hieroglyphs.value_counts()
print("test distinct strings", len(tvc), "test rows sharing string inside test", int((te.hieroglyphs.map(tvc) > 1).sum()))
m = te.hieroglyphs.isin(trs)
print("test rows whose string occurs in train", int(m.sum()), f"{m.mean():.4f}", "words in them", int(te.n_words[m].sum()),
      f"{te.n_words[m].sum()/te.n_words.sum():.4f}", "by test century", te[m].cn.value_counts().to_dict(),
      "n_words mean", round(float(te.n_words[m].mean()), 2))
print("train rows whose string occurs in test", int(tr.hieroglyphs.isin(set(te.hieroglyphs)).sum()),
      "by train century", tr[tr.hieroglyphs.isin(set(te.hieroglyphs))].cn.value_counts().to_dict())

print("\n== 4 vocabulary")
G = re.compile(r"<g>.*?</g>")
TOK = re.compile(r"<g>.*?</g>|.", re.S)


def signs(word):
    return TOK.findall(word)


trw = [w for s in tr.hieroglyphs for w in s.split(" ")]
tew = [w for s in te.hieroglyphs for w in s.split(" ")]
trsig = C(x for w in trw for x in signs(w))
tesig = C(x for w in tew for x in signs(w))
print("train sign tokens", sum(trsig.values()), "distinct (g-code = 1 sign)", len(trsig),
      "distinct unicode chars excl g-codes", sum(1 for k in trsig if not k.startswith("<g>")),
      "distinct g codes", sum(1 for k in trsig if k.startswith("<g>")))
print("train signs seen once", sum(1 for v in trsig.values() if v == 1), "seen <5", sum(1 for v in trsig.values() if v < 5))
un = {k: v for k, v in tesig.items() if k not in trsig}
print("test sign tokens", sum(tesig.values()), "distinct", len(tesig), "distinct unseen in train", len(un), f"{len(un)/len(tesig):.4f}",
      "unseen tokens", sum(un.values()), f"{sum(un.values())/sum(tesig.values()):.4f}")
print("  unseen of which g-codes: distinct", sum(1 for k in un if k.startswith('<g>')), "tokens", sum(v for k, v in un.items() if k.startswith('<g>')))
print("  unseen unicode signs sample", [(k, v) for k, v in sorted(un.items(), key=lambda x: -x[1]) if not k.startswith('<g>')][:15])
print("  test words containing an unseen sign", sum(1 for w in tew if any(x not in trsig for x in signs(w))), f"{sum(1 for w in tew if any(x not in trsig for x in signs(w)))/len(tew):.4f}")
print("  test rows containing an unseen sign", int(te.hieroglyphs.map(lambda s: any(x not in trsig for w in s.split(' ') for x in signs(w))).sum()))
print("variation selectors: train", sum(v for k, v in trsig.items() if 0xFE00 <= ord(k[0]) <= 0xFE0F), "test", sum(v for k, v in tesig.items() if 0xFE00 <= ord(k[0]) <= 0xFE0F))
print("codepoint ranges train:", C((ord(k) >> 8) for k in trsig if not k.startswith("<g>")).most_common())
print("codepoint ranges test:", C((ord(k) >> 8) for k in tesig if not k.startswith("<g>")).most_common())

rows = []
for i, (h, p) in enumerate(zip(tr.hieroglyphs, tr.prediction)):
    for j, (w, gr) in enumerate(zip(h.split(" "), p.split(" "))):
        f = gr.split(";")
        rows.append((i, j, w, f[0], f[1], f[2], f[3], gr))
W = pd.DataFrame(rows, columns=["row", "pos", "w", "tl", "lem", "upos", "gloss", "full"])
W["cn"] = tr.cn.values[W.row.values]
W.to_pickle("solutions/working/egyptian-interlinear-analysis/exp/scout_words.pkl")
FS = re.compile(r"[\\.:\-=]")
feats = C(x for g_ in W.gloss for x in FS.split(g_) if x)
print("train words", len(W), "distinct spellings", W.w.nunique(), "translits", W.tl.nunique(), "lemmas", W.lem.nunique(),
      "UPOS", W.upos.nunique(), "glosses", W.gloss.nunique(), "gloss features", len(feats), "full analyses", W.full.nunique())
print("UPOS counts", W.upos.value_counts().to_dict())
print("top glosses", W.gloss.value_counts().head(25).to_dict())
print("gloss coverage: top10", round(W.gloss.value_counts().head(10).sum() / len(W), 4), "top50", round(W.gloss.value_counts().head(50).sum() / len(W), 4),
      "glosses seen once", int((W.gloss.value_counts() == 1).sum()))
print("features", feats.most_common(70))
print("features seen <5", sum(1 for v in feats.values() if v < 5))
print("spellings seen once", int((W.w.value_counts() == 1).sum()), "lemmas seen once", int((W.lem.value_counts() == 1).sum()))
print("gloss with '=' :", int(W.gloss.str.contains("=").sum()), "gloss starting '-':", int(W.gloss.str.startswith("-").sum()))
print("field has space", int(W.full.str.contains(" ").sum()), "; count per group all 3:", bool((W.full.str.count(";") == 3).all()))

trset = set(trw)
u = np.array([w not in trset for w in tew])
print(f"\nTEST unseen spelling share (tokens) {u.mean():.4f} (published 0.335); distinct test spellings {len(set(tew))}, unseen distinct {len(set(tew)-trset)} ({len(set(tew)-trset)/len(set(tew)):.4f})")
te_cn = np.repeat(te.cn.values, te.n_words.values)
for c in sorted(set(te_cn)):
    print("  test century", c, "words", int((te_cn == c).sum()), "unseen spelling", round(float(u[te_cn == c].mean()), 4))
print("  test rows with zero unseen spellings", int(te.hieroglyphs.map(lambda s: all(w in trset for w in s.split(' '))).sum()))


def rates(held, rest, name):
    hs, rs = W[held], W[rest]
    a = (~hs.w.isin(set(rs.w))).mean()
    b = (~hs.lem.isin(set(rs.lem))).mean()
    c_ = (~hs.lem.str.lower().isin(set(rs.lem.str.lower()))).mean()
    nrow_h = hs.row.nunique()
    reststr = set(tr.hieroglyphs.values[rs.row.unique()])
    hrow = tr.iloc[hs.row.unique()]
    dupr = hrow.hieroglyphs.isin(reststr).mean()
    print(f"  {name}: held rows {nrow_h} words {len(hs)} (mean {len(hs)/nrow_h:.2f}) | train rows {rs.row.nunique()} words {len(rs)} | "
          f"unseen spelling {a:.4f} unseen lemma {b:.4f} (caseless {c_:.4f}) | held rows with string in train part {dupr:.4f}")
    return hs, rs


print("\ncandidate slices (published test: spelling 0.335, lemma 0.097)")
late = W.cn <= 19
hs, rs = rates(late, ~late, "A: <=19th held, rest train")
rates(W.cn <= 18, W.cn > 18, "18th+16th held, rest train")
rates(W.cn <= 18, W.cn > 19, "18th+16th held, >=20th train (19th dropped)")
rates(W.cn == 19, W.cn > 19, "19th held, >=20th train")
rates(W.cn == 19, W.cn != 19, "19th held, all other train")
rates(W.cn.between(20, 22), W.cn > 22, "20-22nd held, earlier-only train")
rates(W.cn.between(20, 22), ~W.cn.between(20, 22), "20-22nd held, all other train")
rates(W.cn.between(20, 22), ~W.cn.between(20, 22) & (W.cn > 19), "20-22nd held, >=23rd train (no later)")
rates(W.cn == 22, W.cn > 22, "22nd held, earlier-only")
rates(W.cn == 23, W.cn > 23, "23rd held, earlier-only")
rates(W.cn == 23, W.cn != 23, "23rd held, all other")
rates(W.cn == 24, W.cn > 24, "24th held, earlier-only")
rates(W.cn == 24, W.cn != 24, "24th held, all other")
rates(W.cn.between(23, 24), ~W.cn.between(23, 24), "23-24th held, all other")
rates(W.cn >= 25, W.cn < 25, ">=25th held, all later train")
rng = np.random.RandomState(0)
rid = rng.rand(len(tr)) < 0.2
rates(pd.Series(rid[W.row.values]), pd.Series(~rid[W.row.values]), "random 20% rows")
# random half of the late slice held out, other half kept in training (how much does having the period help)
half = (tr.cn.values <= 19) & (rng.rand(len(tr)) < 0.5)
rates(pd.Series(half[W.row.values]), pd.Series(~half[W.row.values]), "random half of <=19th held, rest incl other half train")

print("\nper-century unseen spelling vs all other centuries / vs strictly earlier")
for c in sorted(W.cn.unique()):
    h = W[W.cn == c]
    o = set(W[W.cn != c].w); e = set(W[W.cn > c].w)
    print(f"  {c}: words {len(h)} vs-other {(~h.w.isin(o)).mean():.3f} vs-earlier {(~h.w.isin(e)).mean():.3f}")

# test vs train-without-late: how much the late train slice already covers of test spellings
early_set = set(W[~late].w); late_set = set(W[late].w)
print("\ntest words unseen in early-only train", round(float(np.mean([w not in early_set for w in tew])), 4),
      "| unseen in late-only train (<=19th)", round(float(np.mean([w not in late_set for w in tew])), 4),
      "| unseen in 18th-only", round(float(np.mean([w not in set(W[W.cn == 18].w) for w in tew])), 4))

# UPOS / gloss profile of slice vs rest
print("\nUPOS share late slice vs rest")
print(pd.DataFrame({"late": hs.upos.value_counts(normalize=True), "rest": rs.upos.value_counts(normalize=True)}).round(3).to_string())
print("top gloss late:", hs.gloss.value_counts(normalize=True).head(12).round(3).to_dict())
print("top gloss rest:", rs.gloss.value_counts(normalize=True).head(12).round(3).to_dict())
