import sys, re, collections, unicodedata
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
D = "solutions/working/egyptian-interlinear-analysis/dataset/public/"
tr = pd.read_csv(D + "train.csv", encoding="utf-8", keep_default_na=False)
te = pd.read_csv(D + "test.csv", encoding="utf-8", keep_default_na=False)
ss = pd.read_csv(D + "sample_submission.csv", encoding="utf-8", keep_default_na=False)
print("shapes", tr.shape, te.shape, ss.shape)
print("cols", list(tr.columns), list(te.columns), list(ss.columns))
print(tr.dtypes.to_dict())
print("id unique", tr.id.is_unique, te.id.is_unique, "overlap", len(set(tr.id) & set(te.id)))
print("ss ids == test ids in order", (ss.id.values == te.id.values).all())
print("sample ss", ss.prediction.iloc[0][:80])
print("\nHEAD")
for i in range(6):
    r = tr.iloc[i]
    print(r.id, "|", r.latest_century, "|", r.n_words, "|", r.hieroglyphs, "|", r.prediction)
print("test head")
for i in range(4):
    r = te.iloc[i]
    print(r.id, "|", r.latest_century, "|", r.n_words, "|", r.hieroglyphs)

print("\n== 1 century")
print(tr.latest_century.value_counts().to_string())
print(te.latest_century.value_counts().to_string())
tr["w"] = tr.n_words
print("train words per century")
print(tr.groupby("latest_century").n_words.agg(["count", "sum", "mean"]).to_string())
print(te.groupby("latest_century").n_words.agg(["count", "sum", "mean"]).to_string())


def q(name, a):
    a = np.asarray(a)
    print(f"{name}: mean {a.mean():.1f} p50 {np.percentile(a,50):.0f} p90 {np.percentile(a,90):.0f} "
          f"p95 {np.percentile(a,95):.0f} p99 {np.percentile(a,99):.0f} max {a.max()} min {a.min()} sum {a.sum()}")


print("\n== 1 lengths")
q("train n_words", tr.n_words)
q("test n_words", te.n_words)
q("train hiero chars", tr.hieroglyphs.str.len())
q("test hiero chars", te.hieroglyphs.str.len())
q("train hiero bytes", tr.hieroglyphs.map(lambda s: len(s.encode("utf-8"))))
q("test hiero bytes", te.hieroglyphs.map(lambda s: len(s.encode("utf-8"))))
q("train target chars", tr.prediction.str.len())
q("train target bytes", tr.prediction.map(lambda s: len(s.encode("utf-8"))))
tl = tr.prediction.map(lambda s: " ".join(g.split(";")[0] for g in s.split(" ")))
q("train translit-only chars", tl.str.len())
q("train translit-only bytes", tl.map(lambda s: len(s.encode("utf-8"))))
tb = tr.prediction.map(lambda s: len(s.encode("utf-8")))
hb = tr.hieroglyphs.map(lambda s: len(s.encode("utf-8")))
for L in (256, 384, 512, 768, 1024):
    print(f"cap {L}: train input bytes over {(hb > L).mean():.4f} target bytes over {(tb > L).mean():.4f} "
          f"test input over {(te.hieroglyphs.map(lambda s: len(s.encode('utf-8'))) > L).mean():.4f}")
print("target bytes per word mean", tb.sum() / tr.n_words.sum(), "input bytes per word", hb.sum() / tr.n_words.sum())
print("target chars per word", tr.prediction.str.len().sum() / tr.n_words.sum())
# per-word group length
gl = [len(g) for s in tr.prediction for g in s.split(" ")]
q("group chars", gl)
gb = [len(g.encode("utf-8")) for s in tr.prediction for g in s.split(" ")]
q("group bytes", gb)
wl = [len(w.encode("utf-8")) for s in tr.hieroglyphs for w in s.split(" ")]
q("train word bytes", wl)
wl = [len(w.encode("utf-8")) for s in te.hieroglyphs for w in s.split(" ")]
q("test word bytes", wl)
print("n_words hist train", np.bincount(np.minimum(tr.n_words, 60))[:60].tolist())
print("n_words hist test", np.bincount(np.minimum(te.n_words, 60))[:60].tolist())
print("NFC stable: hiero", (tr.hieroglyphs.map(lambda s: unicodedata.normalize("NFC", s)) == tr.hieroglyphs).mean(),
      "pred", (tr.prediction.map(lambda s: unicodedata.normalize("NFC", s)) == tr.prediction).mean(),
      "test", (te.hieroglyphs.map(lambda s: unicodedata.normalize("NFC", s)) == te.hieroglyphs).mean())

print("\n== 2 format")
for name, df in (("train", tr), ("test", te)):
    nw = df.hieroglyphs.map(lambda s: len(s.split(" ")))
    print(name, "n_words == split(' ')", (nw == df.n_words).mean(), "bad", int((nw != df.n_words).sum()))
    print(name, "double space", int(df.hieroglyphs.str.contains("  ").sum()), "lead/trail", int((df.hieroglyphs != df.hieroglyphs.str.strip()).sum()),
          "empty", int((df.hieroglyphs == "").sum()), "other whitespace", int(df.hieroglyphs.str.contains(r"[\t\n\r ]").sum()))
ng = tr.prediction.map(lambda s: len(s.split(" ")))
print("groups == n_words", (ng == tr.n_words).mean(), "bad", int((ng != tr.n_words).sum()))
bad4 = 0; empty = collections.Counter(); tot = 0
for s in tr.prediction:
    for g in s.split(" "):
        tot += 1
        f = g.split(";")
        if len(f) != 4:
            bad4 += 1
            continue
        for k, x in enumerate(f):
            if x == "":
                empty[k] += 1
print("total groups", tot, "not 4 fields", bad4, "empty fields by pos", dict(empty))
print("pred double space", int(tr.prediction.str.contains("  ").sum()), "pred strip", int((tr.prediction != tr.prediction.str.strip()).sum()))

# <g> codes
G = re.compile(r"<g>(.*?)</g>")
for name, df in (("train", tr), ("test", te)):
    codes = collections.Counter(c for s in df.hieroglyphs for c in G.findall(s))
    rows = df.hieroglyphs.str.contains("<g>").sum()
    words = sum(1 for s in df.hieroglyphs for w in s.split(" ") if "<g>" in w)
    print(name, "rows with <g>", int(rows), f"{rows/len(df):.4f}", "words with <g>", words, f"{words/df.n_words.sum():.4f}",
          "code tokens", sum(codes.values()), "distinct", len(codes))
    print("  top", codes.most_common(12))
    print("  code length max", max(len(c) for c in codes), "space inside code", sum(1 for c in codes if " " in c))
    if name == "train":
        trc = set(codes)
    else:
        print("  test codes unseen in train: distinct", len(set(codes) - trc), "tokens", sum(v for c, v in codes.items() if c not in trc), "of", sum(codes.values()))
# non hieroglyph characters in input outside <g>
for name, df in (("train", tr), ("test", te)):
    oc = collections.Counter()
    for s in df.hieroglyphs:
        s2 = G.sub("", s)
        for ch in s2:
            if not (0x13000 <= ord(ch) <= 0x1345F) and ch != " ":
                oc[ch] += 1
    print(name, "non-hieroglyph chars outside <g>:", [(c, hex(ord(c)), n) for c, n in oc.most_common(30)])
    print(name, "examples with <g>:", [s for s in df.hieroglyphs if "<g>" in s][:3])
# target alphabet
tc = collections.Counter(ch for s in tr.prediction for ch in s)
print("target distinct chars", len(tc))
print("target chars", "".join(sorted(tc)), )
print("target rare chars", [(c, hex(ord(c)), n) for c, n in tc.most_common()[-25:]])
